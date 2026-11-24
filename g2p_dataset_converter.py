"""Micro-Level G2P Dataset Converter (Repr. 1 -> Native Homo-GE2PE Repr. 2) & 80/20 Split.

Converts `Dataset for G2P/homograph_dataset_clean_v2.csv` (Repr. 1) into exact
`MahtaFetrat/Homo-GE2PE-Persian` notation (Repr. 2) across all 8 micro-level rules:
  1. Enclitic `-o` (`و`) & ZWNJ compound re-alignment
  2. 16-symbol alphabet & uppercase typo normalization (`A->a`, `a->/`, `S->$`, `C->C`, `Z->;`, `?->@`, `w->v`)
  3. Word-initial glottal `@` injection on vowel-initial tokens
  4. Word-medial hiatus `@` and glide `y` fixes (`Sodeand`, `miAyad`, `bAese`, `niAz`, `xeili`)
  5. Full Kasreh-e-Ezafe `1` tagging on both `-e` and `-ye`
  6. Doubled-letter Tashdid preservation & doubled-vowel typo cleanup
  7. Sentence-final `?` punctuation vs true word-final `ع / ء`
  8. 80% Train / 20% Held-Out Unseen Test stratified split across multi-sense homographs
"""

import collections
import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

from persian_text_normalizer import normalize_persian_text

DEFAULT_CSV_PATH = os.path.join("Dataset for G2P", "homograph_dataset_clean_v2.csv")

_PUNCT_RE = r"[،؛؟.,!\"«»:\s]+"
_VOWELS_REPR2 = set("a/eiou")

_MEDIAL_FIXES: Dict[str, str] = {
    "xeili": "xeyli",
    "niAz": "niyAz",
    "niAze": "niyAze",
    "niAzmand": "niyAzmand",
    "ziAdi": "ziyAdi",
    "ziAd": "ziyAd",
    "keyfiate": "keyfiyyate",
    "keyfiat": "keyfiyyat",
    "miane": "miyAne",
    "miAne": "miyAne",
    "bAes": "bA?es",
    "bAese": "bA?ese",
    "vAqean": "vAqe?an",
    "soAl": "so?Al",
    "soAle": "so?Ale",
    "Sodeand": "Sode?and",
    "kardeand": "karde?and",
    "pardAxteand": "pardAxte?and",
    "dAdeand": "dAde?and",
    "gerefteand": "gerefte?and",
    "neveSteand": "neveSte?and",
    "miAvarad": "mi?Avarad",
    "miAyad": "mi?Ayad",
    "miAmad": "mi?Amad",
    "barrAAq": "barrAq",
    "mobadaal": "mobaddal",
    "pArCeii": "pArCeyi",
    "beeStebAh": "be?eStebAh",
}


def normalize_persian_input(text: str) -> str:
    """Normalize Persian grapheme input for ByT5 G2P."""
    t = normalize_persian_text(
        text,
        use_external=False,
        preserve_zwnj=False,
        expand_numbers_and_dates=True,
        normalize_question_mark=False,
    )
    t = re.sub(r"[ًٌٍَُِّْ]", "", t)
    t = re.sub(_PUNCT_RE, " ", t).strip()
    return t


def convert_word_repr1_to_repr2(pt: str, audit: Dict[str, int] | None = None) -> str:
    """Convert one Repr. 1 dataset word into exact Repr. 2 (`Homo-GE2PE`) notation."""
    if pt in _MEDIAL_FIXES:
        pt = _MEDIAL_FIXES[pt]
        if audit is not None:
            audit["medial_hiatus_or_typo_fixed"] += 1

    fixed: List[str] = []
    for ch in pt:
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

    out: List[str] = []
    for c in fixed:
        if c == "A":
            out.append("a")
        elif c == "a":
            out.append("/")
        elif c == "S":
            out.append("$")
        elif c == "C":
            out.append("C")
        elif c == "Z":
            out.append(";")
        elif c == "?":
            out.append("@")
        else:
            out.append(c)

    res = "".join(out)
    if res and res[0] in _VOWELS_REPR2 and res != "o":
        res = "@" + res
        if audit is not None:
            audit["initial_at_injected"] += 1
    return res


def realign_enclitic_o_and_compounds(
    g_toks: List[str], p_toks: List[str], audit: Dict[str, int] | None = None
) -> List[str]:
    """Recover 1-to-1 alignment when `و` is attached as `-o` to the preceding phoneme token."""
    if len(g_toks) == len(p_toks):
        return p_toks

    realigned: List[str] = []
    g_idx = 0
    for pt in p_toks:
        if (
            g_idx + 1 < len(g_toks)
            and g_toks[g_idx + 1] == "و"
            and len(pt) >= 3
            and pt.endswith("o")
            and not pt.endswith(("oo", "Ao", "uo"))
        ):
            realigned.append(pt[:-1])
            realigned.append("o")
            g_idx += 2
            if audit is not None:
                audit["enclitic_o_split"] += 1
        else:
            realigned.append(pt)
            g_idx += 1
    return realigned


def convert_sentence_repr1_to_repr2(
    grapheme: str, phoneme: str, audit: Dict[str, int] | None = None
) -> Tuple[str, bool]:
    """Convert a full Repr. 1 sentence into Repr. 2 (`Homo-GE2PE`), returning `(repr2_str, is_aligned)`."""
    g_raw = grapheme.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی").strip()
    g_raw_toks = re.sub(r"[،؛.,!\"«»:\s]+", " ", g_raw.replace("\u200c", "")).strip().split()
    g_toks = re.sub(_PUNCT_RE, " ", g_raw.replace("\u200c", "")).strip().split()
    g_split = re.sub(r"[،؛؟.,!\"«»:\u200c\s]+", " ", g_raw).strip().split()
    p_toks = re.sub(r"[،؛.,!\"«»:\s]+", " ", phoneme.strip()).strip().split()

    if len(g_raw_toks) == len(p_toks):
        for idx, (gr, pt) in enumerate(zip(g_raw_toks, p_toks)):
            if gr.endswith(("؟", "?")) and pt.endswith("?"):
                p_toks[idx] = pt[:-1]
                if audit is not None:
                    audit["trailing_question_stripped"] += 1

    # Strip remaining sentence-final '?' on words without final ع/ء/ئ
    if len(g_toks) == len(p_toks):
        for idx, (gw, pt) in enumerate(zip(g_toks, p_toks)):
            if pt.endswith("?") and gw and gw[-1] not in "عءئ":
                p_toks[idx] = pt[:-1]

    p_toks = realign_enclitic_o_and_compounds(g_toks, p_toks, audit=audit)
    aligned_toks = g_toks if len(g_toks) == len(p_toks) else g_split
    is_aligned = len(aligned_toks) == len(p_toks)

    out_toks = [convert_word_repr1_to_repr2(pt, audit=audit) for pt in p_toks]
    if is_aligned:
        for idx, (gw, pw) in enumerate(zip(aligned_toks, out_toks)):
            if (
                len(gw) >= 2
                and gw[-1] not in "هیآاوؤئة"
                and pw.endswith("e")
                and pw not in ("be", "ke", "Ce", "ce", "se", "ne")
            ):
                out_toks[idx] = pw + "1"
                if audit is not None:
                    audit["ezafe_e1_added"] += 1
            elif (
                (gw.endswith(("ه", "ی", "ا", "و", "های")) and not gw.endswith("یه"))
                and pw.endswith("ye")
            ):
                out_toks[idx] = pw + "1"
                if audit is not None:
                    audit["ezafe_ye1_added"] += 1

    return " ".join(tok for tok in out_toks if tok), is_aligned


def canonicalize_repr2_for_eval(phon: str) -> str:
    """Normalize Repr. 2 strings for evaluation, preserving `@` and `C` while stripping only `1` and punctuation."""
    toks = re.sub(_PUNCT_RE, " ", phon).strip().split()
    return " ".join(tok[:-1] if tok.endswith("1") else tok for tok in toks if tok)


def _homograph_lemma_key(word: str) -> str:
    """Normalize a homograph word to its lemma/base key for leak-free word/lemma splitting."""
    w = normalize_persian_text(
        word,
        use_external=False,
        preserve_zwnj=False,
        expand_numbers_and_dates=False,
        normalize_question_mark=False,
    )
    w = re.sub(r"[ًٌٍَُِّْ]", "", w).strip()
    try:
        from hazm import Lemmatizer  # type: ignore

        lemma = Lemmatizer().lemmatize(w)
        if lemma:
            w = lemma.split("#")[0]
    except Exception:
        pass
    return w


def build_80_20_dataset(
    csv_path: str = DEFAULT_CSV_PATH,
    train_ratio: float = 0.80,
    seed: int = 42,
    audit_out_path: str | None = None,
    split_mode: str = "word_lemma",
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    """Build 80% Train / 20% Held-Out Unseen Test split from human-labeled rows only.

    split_mode:
      - 'word_lemma' (default, matches pipeline diagram): splits by homograph word/lemma
        so held-out test homograph words/lemmas are completely unseen during training.
      - 'stratified': splits 80/20 within each (homograph_grapheme, homograph_phoneme) sense.
    """
    rng = random.Random(seed)
    audit: Dict[str, int] = collections.Counter()

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))

    by_hw: Dict[str, Dict[str, List[Dict[str, str]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    seen_graphemes = set()

    for r in raw_rows:
        audit["total_csv_rows"] += 1
        g_orig = (r.get("grapheme") or "").strip()
        p_orig = (r.get("phoneme") or "").strip()
        hw_g = (r.get("homograph_grapheme") or "").strip()
        hw_p = (r.get("homograph_phoneme") or "").strip()
        if not (g_orig and p_orig and hw_g and hw_p):
            audit["skipped_unlabeled_or_incomplete"] += 1
            continue
        if "?systm?" in p_orig or "?Ayn?" in p_orig:
            audit["skipped_noise_placeholder"] += 1
            continue

        g_norm = normalize_persian_input(g_orig)
        if not g_norm or g_norm in seen_graphemes:
            audit["skipped_duplicate_sentence"] += 1
            continue
        seen_graphemes.add(g_norm)

        target_repr2, is_aligned = convert_sentence_repr1_to_repr2(g_orig, p_orig, audit=audit)
        if not is_aligned:
            audit["skipped_unaligned_after_recovery"] += 1
            continue

        hw_repr2 = convert_word_repr1_to_repr2(hw_p)
        item = {
            "grapheme": g_orig,
            "input_text": g_norm,
            "raw_phoneme_repr1": p_orig,
            "target_ge2pe": target_repr2,
            "eval_target": canonicalize_repr2_for_eval(target_repr2),
            "homograph_grapheme": hw_g,
            "homograph_phoneme": hw_p,
            "homograph_eval": canonicalize_repr2_for_eval(hw_repr2),
        }
        by_hw[hw_g][hw_p].append(item)
        audit["clean_converted_sentences"] += 1

    train_items: List[Dict[str, str]] = []
    test_items: List[Dict[str, str]] = []
    if split_mode == "word_lemma":
        by_lemma: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
        for hw in sorted(by_hw.keys()):
            lemma = _homograph_lemma_key(hw)
            for sense in sorted(by_hw[hw].keys()):
                by_lemma[lemma].extend(by_hw[hw][sense])

        lemma_keys = sorted(by_lemma.keys())
        rng.shuffle(lemma_keys)
        total_items = sum(len(v) for v in by_lemma.values())
        target_train = int(round(total_items * train_ratio))

        for lemma in lemma_keys:
            bucket = by_lemma[lemma]
            if len(train_items) < target_train or not train_items:
                train_items.extend(bucket)
            else:
                test_items.extend(bucket)
    else:
        for hw in sorted(by_hw.keys()):
            for sense in sorted(by_hw[hw].keys()):
                bucket = list(by_hw[hw][sense])
                rng.shuffle(bucket)
                if len(bucket) == 1:
                    train_items.extend(bucket)
                else:
                    n_train = max(1, int(round(len(bucket) * train_ratio)))
                    if n_train == len(bucket):
                        n_train -= 1
                    train_items.extend(bucket[:n_train])
                    test_items.extend(bucket[n_train:])

    rng.shuffle(train_items)
    rng.shuffle(test_items)

    # 8 showcase items (4 seen in train, 4 held-out in test) across distinct homographs
    showcase_items: List[Dict[str, str]] = []
    for hw in ("ابر", "اجر", "اره", "سمت"):
        senses = sorted(by_hw.get(hw, {}).keys())
        if len(senses) >= 2:
            s_tr = next((x for x in train_items if x["homograph_grapheme"] == hw and x["homograph_phoneme"] == senses[0]), None)
            s_te = next((x for x in test_items if x["homograph_grapheme"] == hw and x["homograph_phoneme"] == senses[1]), None)
            if not s_te:
                s_te = next((x for x in test_items if x["homograph_grapheme"] == hw), None)
            if s_tr:
                showcase_items.append({**s_tr, "split_tag": "TRAIN (Seen)"})
            if s_te:
                showcase_items.append({**s_te, "split_tag": "TEST (Held-Out)"})

    summary = {
        "csv_path": csv_path,
        "split_mode": split_mode,
        "train_count": len(train_items),
        "test_unseen_count": len(test_items),
        "showcase_count": len(showcase_items),
        "audit_counts": dict(audit),
    }
    if audit_out_path:
        os.makedirs(os.path.dirname(audit_out_path) or ".", exist_ok=True)
        with open(audit_out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return train_items, test_items, showcase_items, summary
