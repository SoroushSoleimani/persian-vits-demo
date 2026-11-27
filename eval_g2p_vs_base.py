"""
G2P-Level Evaluation Benchmark: Base G2P vs Upgraded G2P (Normalization + Stress/Homograph).

Computes standard G2P metrics (PER, WER, Word Accuracy, Sentence Accuracy, Stressed Word Accuracy,
Homograph Accuracy) on a curated synthetic Persian benchmark (no private dataset rows) and prints
side-by-side Before vs After IPA outputs.
"""

import json
import os
import re
from typing import Any, Dict, List

from finetune_homo_ge2pe_80pct import compute_delta_vs_base, levenshtein
from g2p_persian import PersianG2PProsody

# Curated synthetic Persian G2P benchmark (phoneme homographs, stress homographs, tanvin, tashdid)
G2P_BENCHMARK: List[Dict[str, str]] = [
    {
        "id": "1",
        "category": "Stress Homograph (ماهی: fish)",
        "text": "ماهی قرمز بود.",
        "target_word_ipa": "mɑhˈi",
        "ref_ipa": "mɑhˈi qermez bˈud",
    },
    {
        "id": "2",
        "category": "Stress Homograph (ماهی: monthly)",
        "text": "ماهی دو بار رفت.",
        "target_word_ipa": "mˈɑhi",
        "ref_ipa": "mˈɑhi do bɑr rˈaft",
    },
    {
        "id": "3",
        "category": "Stress Homograph (مردی: a man)",
        "text": "مردی از دور رفت.",
        "target_word_ipa": "mˈardi",
        "ref_ipa": "mˈardi az dur rˈaft",
    },
    {
        "id": "4",
        "category": "Stress Homograph (مردی: manliness)",
        "text": "از مردی و محبت او",
        "target_word_ipa": "mardˈi",
        "ref_ipa": "az mardˈi o mohabbˈat u",
    },
    {
        "id": "5",
        "category": "Stress Homograph (ولی: conjunction)",
        "text": "ولی او خوب بود.",
        "target_word_ipa": "vˈali",
        "ref_ipa": "vˈali u xˈub bˈud",
    },
    {
        "id": "6",
        "category": "Stress Homograph (ولی: guardian)",
        "text": "ولی فقیه",
        "target_word_ipa": "valˈi",
        "ref_ipa": "valˈije faqih",
    },
    {
        "id": "7",
        "category": "Phoneme Homograph (کرم: karam vs kerm)",
        "text": "کرم مرد بزرگ",
        "target_word_ipa": "karˈame",
        "ref_ipa": "karˈame mˈarde bozˈorɡ",
    },
    {
        "id": "8",
        "category": "Phoneme Homograph (کرم: worm)",
        "text": "کرم کوچک روی برگ درخت بود.",
        "target_word_ipa": "kˈerme",
        "ref_ipa": "kˈerme kutʃˈak rˈuje bˈarɡe derˈaxt bˈud",
    },
    {
        "id": "9",
        "category": "Phoneme Homograph (مهر: mohr vs mehr)",
        "text": "مهر استاد دانشگاه",
        "target_word_ipa": "mˈohre",
        "ref_ipa": "mˈohre ostˈɑde dɑneʃɡˈɑh",
    },
    {
        "id": "10",
        "category": "Tanvin + Tashdid",
        "text": "حتما با دقت رفت.",
        "target_word_ipa": "hˈatman",
        "ref_ipa": "hˈatman bɑ dˈeqqat rˈaft",
    },
]


def _clean_ipa(s: str, keep_stress: bool = True) -> str:
    s = s.replace("^", "").replace("$", "").replace("_", "")
    s = re.sub(r"[.,!?;:،؛؟]", " ", s)
    if not keep_stress:
        s = s.replace("ˈ", "").replace("ˌ", "")
    return re.sub(r"\s+", " ", s).strip()


def _base_g2p_ipa(g2p: PersianG2PProsody, text: str) -> str:
    """Base rule-only G2P (no context homograph or stress-homograph disambiguation)."""
    syms, _, _ = g2p.text_to_aligned_streams(text, intersperse_blank=False)
    return _clean_ipa("".join(syms), keep_stress=True)


def _upgraded_g2p_ipa(g2p: PersianG2PProsody, text: str) -> str:
    """Upgraded G2P with Persian normalization + context/stress homograph disambiguation."""
    syms, _, _, _, _ = g2p.text_to_aligned_streams_with_context(
        text, intersperse_blank=False, prefer_homo_ge2pe=False
    )
    return _clean_ipa("".join(syms), keep_stress=True)


def evaluate_ipa_predictions(
    items: List[Dict[str, str]], preds: List[str]
) -> Dict[str, Any]:
    total_chars = 0
    total_char_edits = 0
    total_st_chars = 0
    total_st_edits = 0
    total_words = 0
    total_word_edits = 0
    exact_words = 0
    stressed_exact_words = 0
    exact_sentences = 0
    hw_hits = 0
    details: List[Dict[str, Any]] = []

    for item, hyp_raw in zip(items, preds):
        ref_st = _clean_ipa(item["ref_ipa"], keep_stress=True)
        hyp_st = _clean_ipa(hyp_raw, keep_stress=True)
        ref = _clean_ipa(ref_st, keep_stress=False)
        hyp = _clean_ipa(hyp_st, keep_stress=False)

        c_edits = levenshtein(ref, hyp)
        st_edits = levenshtein(ref_st, hyp_st)
        ref_w, hyp_w = ref.split(), hyp.split()
        w_edits = levenshtein(ref_w, hyp_w)

        total_char_edits += c_edits
        total_chars += max(1, len(ref))
        total_st_edits += st_edits
        total_st_chars += max(1, len(ref_st))
        total_word_edits += w_edits
        total_words += max(1, len(ref_w))

        for rw, hw in zip(ref_w, hyp_w):
            if rw == hw:
                exact_words += 1
        for rw_s, hw_s in zip(ref_st.split(), hyp_st.split()):
            if rw_s == hw_s:
                stressed_exact_words += 1
        if ref == hyp:
            exact_sentences += 1

        target_ipa = item["target_word_ipa"]
        hw_hit = target_ipa in hyp_st
        if hw_hit:
            hw_hits += 1

        details.append(
            {
                "id": item["id"],
                "category": item["category"],
                "text": item["text"],
                "ref_ipa": ref_st,
                "pred_ipa": hyp_st,
                "per": round(c_edits / max(1, len(ref)), 4),
                "wer": round(w_edits / max(1, len(ref_w)), 4),
                "target_hit": hw_hit,
            }
        )

    return {
        "count": len(items),
        "per": round(total_char_edits / max(1, total_chars), 4),
        "wer": round(total_word_edits / max(1, total_words), 4),
        "word_exact_acc": round(exact_words / max(1, total_words), 4),
        "sentence_exact_acc": round(exact_sentences / max(1, len(items)), 4),
        "stressed_per": round(total_st_edits / max(1, total_st_chars), 4),
        "stressed_word_acc": round(stressed_exact_words / max(1, total_words), 4),
        "homograph_acc": round(hw_hits / max(1, len(items)), 4),
        "homograph_hits": hw_hits,
        "items": details,
    }


def run_benchmark(save_json_path: str = os.path.join("output", "g2p_vs_base_report.json")) -> Dict[str, Any]:
    g2p = PersianG2PProsody()
    base_preds = [_base_g2p_ipa(g2p, it["text"]) for it in G2P_BENCHMARK]
    up_preds = [_upgraded_g2p_ipa(g2p, it["text"]) for it in G2P_BENCHMARK]

    base_metrics = evaluate_ipa_predictions(G2P_BENCHMARK, base_preds)
    up_metrics = evaluate_ipa_predictions(G2P_BENCHMARK, up_preds)
    deltas = compute_delta_vs_base(base_metrics, up_metrics)

    report = {
        "base_g2p": {k: v for k, v in base_metrics.items() if k != "items"},
        "upgraded_g2p": {k: v for k, v in up_metrics.items() if k != "items"},
        "delta_vs_base": deltas,
        "samples": [
            {
                "id": item["id"],
                "category": item["category"],
                "text": item["text"],
                "target_word_ipa": item["target_word_ipa"],
                "base_ipa": base_metrics["items"][i]["pred_ipa"],
                "base_hit": base_metrics["items"][i]["target_hit"],
                "upgraded_ipa": up_metrics["items"][i]["pred_ipa"],
                "upgraded_hit": up_metrics["items"][i]["target_hit"],
            }
            for i, item in enumerate(G2P_BENCHMARK)
        ],
    }

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path) or ".", exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    rep = run_benchmark()
    print("=== G2P Evaluation Summary (Base vs Upgraded) ===")
    print(json.dumps({"base_g2p": rep["base_g2p"], "upgraded_g2p": rep["upgraded_g2p"], "delta_vs_base": rep["delta_vs_base"]}, ensure_ascii=False, indent=2))
    print("\n=== Sample Outputs (Base vs Upgraded) ===")
    for s in rep["samples"]:
        print(f"[{s['id']}] {s['category']} | {s['text']}")
        print(f"    Base    (hit={s['base_hit']}): {s['base_ipa']}")
        print(f"    Upgraded(hit={s['upgraded_hit']}): {s['upgraded_ipa']}")
