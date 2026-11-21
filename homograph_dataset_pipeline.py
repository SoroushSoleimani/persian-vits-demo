"""Block D: Homograph Dataset Pipeline (Repr. 1 -> Repr. 2, Word/Lemma Split & Extra Real TTS Test Set)."""

import collections
import csv
import os
import random
import re
from typing import Dict, List, Tuple

DEFAULT_CSV_PATH = os.path.join("Dataset for G2P", "homograph_dataset_clean_v2.csv")

EXTRA_REAL_TTS_TEST_SET: List[Dict[str, str]] = [
    {
        "id": "tts_1",
        "text": "تاریخ جلسه ۱۴۰۳/۰۸/۲۵ با د. رضایی در سمت ریاست برگزار می‌شود؟",
        "focus": "Date + Abbreviation (د.) + Homograph (semat) + Question (؟)",
    },
    {
        "id": "tts_2",
        "text": "مدیر جدید ۲۵٪ بودجه را با دقت فراوان بررسی کرد.",
        "focus": "Percentage (۲۵٪) + Tashdid (deqqat) + ZWNJ/Ezafe",
    },
    {
        "id": "tts_3",
        "text": "معلم مدرسه بیش از ۱۲۰ صفحه کتاب را می‌خواند.",
        "focus": "Number (۱۲۰) + Tashdid+Ezafe (mo'allem-e) + ZWNJ (می‌خواند)",
    },
    {
        "id": "tts_4",
        "text": "تیغه اره در مسافت ۵ km بسیار تیز می‌ماند.",
        "focus": "Homograph (arre) + Number + Abbreviation (km) + ZWNJ",
    },
    {
        "id": "tts_5",
        "text": "باد از سمت دریا با سرعت ۳.۵ برابر به سوی ساحل می‌وزید.",
        "focus": "Homograph (samt) + Decimal (۳.۵) + ZWNJ (می‌وزید)",
    },
]


def repr1_word_to_repr2(pt: str) -> str:
    """Convert Repr. 1 (Pinglish dataset) word into Repr. 2 (Homo-GE2PE-Persian)."""
    out: List[str] = []
    for ch in pt:
        if ch == "A":
            out.append("a")
        elif ch == "a":
            out.append("/")
        elif ch == "S":
            out.append("$")
        elif ch in ("C", "c"):
            out.append("C")
        elif ch == "Z":
            out.append(";")
        elif ch == "?":
            out.append("@")
        else:
            out.append(ch)
    return "".join(out)


def convert_dataset_phoneme_to_repr2(grapheme: str, phoneme: str) -> str:
    """Convert full Repr. 1 sentence transcription into Repr. 2 (`Homo-GE2PE`) notation."""
    fixed: List[str] = []
    for ch in phoneme.strip():
        if ch in ("A", "S", "C", "Z"):
            fixed.append(ch)
        elif ch == "c":
            fixed.append("C")
        elif ch == "w":
            fixed.append("v")
        elif "A" <= ch <= "Z":
            fixed.append(ch.lower())
        elif ch == ",":
            continue
        else:
            fixed.append(ch)
    s = "".join(fixed)

    g_raw = grapheme.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی").strip()
    g_raw_toks = re.sub(r"[،؛.,!\"«»:\s]+", " ", g_raw.replace("\u200c", "")).strip().split()
    g_toks = re.sub(r"[،؛؟.,!\"«»:\s]+", " ", g_raw.replace("\u200c", "")).strip().split()
    p_toks = re.sub(r"[،؛؟.,!\"«»:\s]+", " ", s).strip().split()

    if len(g_raw_toks) == len(p_toks):
        for idx, (gr, pt) in enumerate(zip(g_raw_toks, p_toks)):
            if gr.endswith(("؟", "?")) and pt.endswith("?"):
                p_toks[idx] = pt[:-1]

    out_toks = [repr1_word_to_repr2(pt) for pt in p_toks]
    if len(g_toks) == len(out_toks):
        for idx, (gw, pw) in enumerate(zip(g_toks, out_toks)):
            if (
                len(gw) >= 2
                and gw[-1] not in "هیآاوؤئة"
                and pw.endswith("e")
                and pw not in ("be", "ke", "Ce", "ce", "se", "ne")
            ):
                out_toks[idx] = pw + "1"
            elif gw.endswith("های") and pw.endswith("haye"):
                out_toks[idx] = pw + "1"
    return " ".join(tok for tok in out_toks if tok)


def load_homograph_dataset_by_word_lemma(
    csv_path: str = DEFAULT_CSV_PATH,
    train_limit: int = 2400,
    test_limit: int = 300,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load homograph rows and perform word/lemma-stratified train/test split + real TTS eval set."""
    rng = random.Random(seed)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))

    by_hw: Dict[str, Dict[str, List[Dict[str, str]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    seen_sentences = set()

    for r in raw_rows:
        g_orig = (r.get("grapheme") or "").strip()
        p_orig = (r.get("phoneme") or "").strip()
        hw_g = (r.get("homograph_grapheme") or "").strip()
        hw_p = (r.get("homograph_phoneme") or "").strip()
        if not (g_orig and p_orig and hw_g and hw_p):
            continue

        g_joined = re.sub(r"[،؛؟.,!\"«»:\s]+", " ", g_orig.replace("\u200c", "")).strip().split()
        g_split = re.sub(r"[،؛؟.,!\"«»:\u200c\s]+", " ", g_orig).strip().split()
        ph_toks = re.sub(r"[،؛؟.,!\"«»:\s]+", " ", p_orig).strip().split()
        if len(g_joined) != len(ph_toks) and len(g_split) != len(ph_toks):
            continue

        g_key = re.sub(r"\s+", " ", g_orig.replace("\u200c", " ")).strip()
        if g_key in seen_sentences:
            continue
        seen_sentences.add(g_key)

        target_repr2 = convert_dataset_phoneme_to_repr2(g_orig, p_orig)
        hw_repr2 = repr1_word_to_repr2(
            "".join(
                c if c in ("A", "S", "C", "Z") else ("v" if c == "w" else c.lower())
                for c in hw_p
            )
        )
        by_hw[hw_g][hw_p].append(
            {
                "grapheme": g_orig,
                "raw_phoneme_repr1": p_orig,
                "target_repr2": target_repr2,
                "homograph_grapheme": hw_g,
                "homograph_phoneme_repr1": hw_p,
                "homograph_repr2": hw_repr2,
            }
        )

    multi_hws = [hw for hw, senses in by_hw.items() if len(senses) >= 2]
    rng.shuffle(multi_hws)
    train_items: List[Dict[str, str]] = []
    test_items: List[Dict[str, str]] = []

    for hw in multi_hws:
        for sk in sorted(by_hw[hw].keys()):
            bucket = list(by_hw[hw][sk])
            rng.shuffle(bucket)
            n_test = 1 if len(bucket) >= 3 else 0
            if len(bucket) >= 10:
                n_test = max(2, int(round(len(bucket) * 0.2)))
            test_items.extend(bucket[:n_test])
            train_items.extend(bucket[n_test:])

    rng.shuffle(train_items)
    rng.shuffle(test_items)
    return train_items[:train_limit], test_items[:test_limit], list(EXTRA_REAL_TTS_TEST_SET)
