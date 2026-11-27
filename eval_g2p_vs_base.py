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


def save_comparison_chart_svg(report: Dict[str, Any], svg_path: str) -> None:
    """Generate a crisp pure-SVG grouped bar chart comparing Base vs Upgraded G2P metrics."""
    base = report["base_g2p"]
    up = report["upgraded_g2p"]
    metrics = [
        ("PER (Lower)", base["per"] * 100.0, up["per"] * 100.0),
        ("WER (Lower)", base["wer"] * 100.0, up["wer"] * 100.0),
        ("Stressed PER (Lower)", base["stressed_per"] * 100.0, up["stressed_per"] * 100.0),
        ("Word Acc (Higher)", base["word_exact_acc"] * 100.0, up["word_exact_acc"] * 100.0),
        ("Stressed Acc (Higher)", base["stressed_word_acc"] * 100.0, up["stressed_word_acc"] * 100.0),
        ("Homograph Acc (Higher)", base["homograph_acc"] * 100.0, up["homograph_acc"] * 100.0),
    ]

    width, height = 960, 440
    left, right, top, bottom = 70, 30, 70, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(metrics)
    bar_w = 34

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="480" y="34" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="19" font-weight="700" fill="#111827">Persian G2P Evaluation: Base vs Upgraded (PER / WER / Accuracy %)</text>',
        '<rect x="330" y="46" width="14" height="12" rx="2" fill="#ef4444"/>',
        '<text x="350" y="57" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#374151">Base G2P</text>',
        '<rect x="450" y="46" width="14" height="12" rx="2" fill="#2563eb"/>',
        '<text x="470" y="57" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#374151">Upgraded G2P (Norm + Stress/Homograph)</text>',
    ]

    for tick in range(0, 101, 20):
        y = top + plot_h - (tick / 100.0) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7280">{tick}%</text>')

    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#9ca3af" stroke-width="1.5"/>')

    for i, (label, b_val, u_val) in enumerate(metrics):
        cx = left + group_w * (i + 0.5)
        b_h = (b_val / 100.0) * plot_h
        u_h = (u_val / 100.0) * plot_h
        bx = cx - bar_w - 4
        ux = cx + 4
        by = top + plot_h - b_h
        uy = top + plot_h - u_h

        lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{b_h:.1f}" rx="4" fill="#ef4444"/>')
        lines.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="600" fill="#b91c1c">{b_val:.1f}%</text>')

        lines.append(f'<rect x="{ux:.1f}" y="{uy:.1f}" width="{bar_w}" height="{u_h:.1f}" rx="4" fill="#2563eb"/>')
        lines.append(f'<text x="{ux + bar_w / 2:.1f}" y="{uy - 6:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="600" fill="#1d4ed8">{u_val:.1f}%</text>')

        lines.append(f'<text x="{cx:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="600" fill="#1f2937">{label}</text>')

    lines.append("</svg>")
    os.makedirs(os.path.dirname(svg_path) or ".", exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_comparison_markdown(report: Dict[str, Any], md_path: str, chart_rel_path: str = "g2p_vs_base_chart.svg") -> None:
    """Save a Persian Markdown evaluation report with metrics table, chart link, and sample outputs."""
    b = report["base_g2p"]
    u = report["upgraded_g2p"]
    d = report["delta_vs_base"]

    md_lines = [
        "# گزارش ارزیابی G2P فارسی (مقایسه حالت پایه در برابر حالت ارتقایافته)",
        "",
        f"![G2P Evaluation Chart]({chart_rel_path})",
        "",
        "## ۱) مقایسه عددی معیارهای ارزیابی (Base vs Upgraded)",
        "",
        "| معیار | حالت پایه (Base G2P) | حالت ارتقایافته (Upgraded G2P) | تغییر نسبت به پایه | توضیح |",
        "| :--- | :---: | :---: | :---: | :--- |",
        f"| **PER** (خطای واجی) | `{b['per']:.4f}` ({b['per']*100:.2f}%) | `{u['per']:.4f}` ({u['per']*100:.2f}%) | `{d['delta_per']:+.4f}` | کمتر بهتر است؛ کاهش خطای سطح واج |",
        f"| **WER** (خطای کلمه‌ای) | `{b['wer']:.4f}` ({b['wer']*100:.2f}%) | `{u['wer']:.4f}` ({u['wer']*100:.2f}%) | `{d['delta_wer']:+.4f}` | کمتر بهتر است؛ کاهش خطای سطح کلمه |",
        f"| **Stressed PER** | `{b['stressed_per']:.4f}` ({b['stressed_per']*100:.2f}%) | `{u['stressed_per']:.4f}` ({u['stressed_per']*100:.2f}%) | `{d['delta_stressed_per']:+.4f}` | خطای واجی با احتساب جای دقیق تکیه (`ˈ`) |",
        f"| **Word Exact Accuracy** | `{b['word_exact_acc']*100:.2f}%` | `{u['word_exact_acc']*100:.2f}%` | `{d['delta_word_exact_acc']*100:+.2f}%` | درصد کلمات با تلفظ کاملاً منطبق |",
        f"| **Stressed Word Accuracy** | `{b['stressed_word_acc']*100:.2f}%` | `{u['stressed_word_acc']*100:.2f}%` | `{d['delta_stressed_word_acc']*100:+.2f}%` | درصد کلمات منطبق با احتساب جای تکیه |",
        f"| **Sentence Exact Accuracy** | `{b['sentence_exact_acc']*100:.2f}%` | `{u['sentence_exact_acc']*100:.2f}%` | `{d['delta_sentence_exact_acc']*100:+.2f}%` | درصد جملات کاملاً بدون خطا |",
        f"| **Homograph & Stress Acc** | `{b['homograph_acc']*100:.1f}%` ({b['homograph_hits']}/{b['count']}) | `{u['homograph_acc']*100:.1f}%` ({u['homograph_hits']}/{u['count']}) | `{d['delta_homograph_acc']*100:+.1f}%` | دقت رفع ابهام هموگراف‌های واجی و تکیه‌ای |",
        "",
        "## ۲) مقایسه نمونه‌های خروجی (Base در برابر Upgraded)",
        "",
        "| # | دسته | جمله فارسی | هدف مورد انتظار | خروجی Base G2P | خروجی Upgraded G2P | وضعیت |",
        "| :---: | :--- | :--- | :---: | :--- | :--- | :---: |",
    ]
    for s in report["samples"]:
        status = "اصلاح شد" if (not s["base_hit"] and s["upgraded_hit"]) else ("هر دو درست" if s["upgraded_hit"] else "نیاز به بهبود")
        md_lines.append(
            f"| {s['id']} | {s['category']} | {s['text']} | `{s['target_word_ipa']}` | `{s['base_ipa']}` | `{s['upgraded_ipa']}` | {status} |"
        )

    md_lines.extend([
        "",
        "## ۳) تحلیل کوتاه نتایج",
        "- **هموگراف‌های تکیه‌ای (Stress Homographs):** در جفت‌های «ماهی» (`mɑhˈi` آبزی در برابر `mˈɑhi` ماهانه)، «مردی» (`mˈardi` یک مرد در برابر `mardˈi` مردانگی) و «ولی» (`vˈali` حرف ربط در برابر `valˈi` سرپرست)، حالت پایه فاقد تشخیص کانتکستی تکیه است؛ در حالی که حالت ارتقایافته با مارکر `'` جای `ˈ` و تگ `PROSODY_TAGS['STRESS']` را دقیق تنظیم می‌کند.",
        "- **هموگراف‌های واجی و کسره اضافه:** در «کرم مرد بزرگ» (`karˈame`) در برابر «کرم کوچک» (`kˈerme`) و «مهر استاد» (`mˈohre`)، حالت ارتقایافته هم مصوت کوتاه و هم کسره اضافه (`-e`) را صحیح تولید می‌کند.",
        "- **تشدید و تنوین:** در «حتما» و «دقت»، تنوین (`-an`) و تشدید (`q1q1`) حفظ می‌شوند و به تگ `TASHDID` در VITS منتقل می‌گردند.",
    ])

    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def run_benchmark(
    save_json_path: str = os.path.join("output", "g2p_vs_base_report.json"),
    save_svg_path: str = os.path.join("output", "g2p_vs_base_chart.svg"),
    save_md_path: str = os.path.join("output", "g2p_vs_base_report.md"),
) -> Dict[str, Any]:
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
    if save_svg_path:
        save_comparison_chart_svg(report, save_svg_path)
    if save_md_path:
        save_comparison_markdown(report, save_md_path, chart_rel_path=os.path.basename(save_svg_path) if save_svg_path else "")

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
