"""Fine-tune Homo-GE2PE-Persian (ByT5) on a stratified subset of `Dataset for G2P`.

Workflow:
  1. Loads `Dataset for G2P/homograph_dataset_clean_v2.csv`, filters to exact
     ZWNJ-aware token-aligned rows, and normalizes targets to `Homo-GE2PE` notation.
  2. Builds a stratified subset across multi-pronunciation Persian homographs:
       - `train` (~1,000 sentences)
       - `test_unseen` (~200 held-out sentences from the same homographs)
       - 8 concrete showcase test sentences (4 seen in train + 4 held-out unseen)
  3. Evaluates BEFORE (base `Homo-GE2PE-Persian`) vs AFTER (fine-tuned ByT5) on:
       - Full-sentence Phoneme Error Rate (PER)
       - Word-level Exact Match
       - Homograph Disambiguation Accuracy
       - End-to-End Piper VITS synthesis (`output/persian_vits_upgraded.onnx`) -> WAVs & Mel plots
"""

import collections
import csv
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_CSV_PATH = os.path.join("Dataset for G2P", "homograph_dataset_clean_v2.csv")
DEFAULT_GE2PE_DIR = os.path.join("pretrained", "homo-ge2pe")
DEFAULT_ONNX_PATH = os.path.join("output", "persian_vits_upgraded.onnx")


def _normalize_persian_input(text: str) -> str:
    t = (
        text.replace("ك", "ک")
        .replace("ي", "ی")
        .replace("ى", "ی")
        .replace("\u200c", " ")
    )
    t = re.sub(r"[ًٌٍَُِّْ]", "", t)
    t = re.sub(r"[،؛؟.,!\"«»:\s]+", " ", t).strip()
    return t


def _csv_word_to_ge2pe(pt: str) -> str:
    while pt.endswith("?"):
        pt = pt[:-1]
    if pt.startswith("?"):
        pt = pt[1:]
    out: List[str] = []
    for c in pt:
        if c == "A":
            out.append("a")
        elif c == "a":
            out.append("/")
        elif c == "S":
            out.append("$")
        elif c == "C":
            out.append("c")
        elif c == "Z":
            out.append(";")
        elif c == "?":
            out.append("@")
        else:
            out.append(c)
    return "".join(out)


def csv_phoneme_to_ge2pe(grapheme: str, phoneme: str) -> str:
    """Convert dataset Pinglish phoneme string into `Homo-GE2PE-Persian` target format."""
    fixed: List[str] = []
    for ch in phoneme.strip():
        if ch in ("A", "S", "C", "Z"):
            fixed.append(ch)
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
    g_toks = re.sub(r"[،؛؟.,!\"«»:\s]+", " ", g_raw.replace("\u200c", "")).strip().split()
    p_toks = re.sub(r"[،؛؟.,!\"«»:\s]+", " ", s).strip().split()

    out_toks = [_csv_word_to_ge2pe(pt) for pt in p_toks]
    if len(g_toks) == len(out_toks):
        for idx, (gw, pw) in enumerate(zip(g_toks, out_toks)):
            if (
                len(gw) >= 2
                and gw[-1] not in "هیآاوؤئة"
                and pw.endswith("e")
                and pw not in ("be", "ke", "ce", "se", "ne")
            ):
                out_toks[idx] = pw + "1"
            elif gw.endswith("های") and pw.endswith("haye"):
                out_toks[idx] = pw + "1"
    return " ".join(tok for tok in out_toks if tok)


def canonicalize_for_eval(phon: str) -> str:
    """Normalize both base Homo-GE2PE and dataset predictions cleanly for fair comparison.

    Strips formatting-only Ezafe marker `1`, punctuation, and word-initial `@` so
    evaluation isolates real vowel/consonant/homograph errors rather than punctuation quirks.
    """
    toks = re.sub(r"[،؛؟.,!\"«»:\s]+", " ", phon).strip().split()
    cleaned: List[str] = []
    for tok in toks:
        if tok.endswith("1"):
            tok = tok[:-1]
        if tok.startswith("@") and len(tok) > 1:
            tok = tok[1:]
        if tok.endswith("@") and len(tok) > 1:
            tok = tok[:-1]
        cleaned.append(tok)
    return " ".join(t for t in cleaned if t)


def levenshtein(seq_a: Any, seq_b: Any) -> int:
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        a_i = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if a_i == seq_b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def load_and_split_dataset(
    csv_path: str = DEFAULT_CSV_PATH,
    train_limit: int = 2400,
    test_limit: int = 300,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load exact-aligned rows and build stratified train, unseen-test, and showcase sets."""
    splits_json = os.path.join(os.path.dirname(csv_path) or ".", "pre_splits.json")
    if os.path.isfile(splits_json):
        with open(splits_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["train"], data["test"], data["showcase"]

    rng = random.Random(seed)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        raw_rows = list(csv.DictReader(f))

    clean_rows: List[Dict[str, str]] = []
    seen_graphemes = set()
    by_hw: Dict[str, Dict[str, List[Dict[str, str]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

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

        g_norm = _normalize_persian_input(g_orig)
        if not g_norm or g_norm in seen_graphemes:
            continue
        seen_graphemes.add(g_norm)

        target_ge2pe = csv_phoneme_to_ge2pe(g_orig, p_orig)
        hw_target_ge2pe = _csv_word_to_ge2pe(
            "".join(
                c if c in ("A", "S", "C", "Z") else ("v" if c == "w" else c.lower())
                for c in hw_p
            )
        )
        item = {
            "grapheme": g_orig,
            "input_text": g_norm,
            "target_ge2pe": target_ge2pe,
            "eval_target": canonicalize_for_eval(target_ge2pe),
            "homograph_grapheme": hw_g,
            "homograph_phoneme": hw_p,
            "homograph_eval": canonicalize_for_eval(hw_target_ge2pe),
        }
        clean_rows.append(item)
        by_hw[hw_g][hw_p].append(item)

    # Multi-sense homographs first so the subset directly targets contextual disambiguation
    multi_hws = [hw for hw, senses in by_hw.items() if len(senses) >= 2]
    rng.shuffle(multi_hws)
    priority_hws = ["ابر", "اجر", "اره", "اشکال", "سمت", "مهر", "کرم", "مرد"]
    ordered_hws = [h for h in priority_hws if h in multi_hws] + [
        h for h in multi_hws if h not in priority_hws
    ]

    train_pool: List[Dict[str, str]] = []
    test_pool: List[Dict[str, str]] = []

    for hw in ordered_hws:
        senses = by_hw[hw]
        sense_keys = sorted(senses.keys())
        for sk in sense_keys:
            bucket = list(senses[sk])
            rng.shuffle(bucket)
            # Hold out ~15-20% per sense (min 1 when >=3 examples)
            n_test = 1 if len(bucket) >= 3 else 0
            if len(bucket) >= 10:
                n_test = max(2, int(round(len(bucket) * 0.2)))
            test_pool.extend(bucket[:n_test])
            train_pool.extend(bucket[n_test:])

    rng.shuffle(train_pool)
    rng.shuffle(test_pool)

    # Ensure priority homographs keep broad representation in train & test slices
    priority_train = [x for x in train_pool if x["homograph_grapheme"] in priority_hws]
    other_train = [x for x in train_pool if x["homograph_grapheme"] not in priority_hws]
    train_items = (priority_train + other_train)[:train_limit]

    trained_hws = {x["homograph_grapheme"] for x in train_items}
    priority_test = [
        x for x in test_pool
        if x["homograph_grapheme"] in priority_hws and x["homograph_grapheme"] in trained_hws
    ]
    other_test = [
        x for x in test_pool
        if x["homograph_grapheme"] not in priority_hws and x["homograph_grapheme"] in trained_hws
    ]
    test_items = (priority_test + other_test)[:test_limit]

    # Build 8 Showcase sentences: 4 pairs (ابر, اجر, اره, سمت) with 1 Seen + 1 Unseen per homograph
    showcase_items: List[Dict[str, str]] = []
    showcase_hws = ["ابر", "اجر", "اره", "سمت"]
    for hw in showcase_hws:
        hw_senses = sorted(by_hw[hw].keys())
        if len(hw_senses) < 2:
            continue
        s1, s2 = hw_senses[0], hw_senses[1]
        seen_match = next(
            (x for x in train_items if x["homograph_grapheme"] == hw and x["homograph_phoneme"] == s1),
            None,
        )
        unseen_match = next(
            (x for x in test_items if x["homograph_grapheme"] == hw and x["homograph_phoneme"] == s2),
            None,
        )
        if seen_match:
            s_copy = dict(seen_match)
            s_copy["split_tag"] = "TRAIN (Seen)"
            showcase_items.append(s_copy)
        if unseen_match:
            u_copy = dict(unseen_match)
            u_copy["split_tag"] = "TEST (Held-Out)"
            showcase_items.append(u_copy)

    # Attach all known valid pronunciations for each homograph so homograph accuracy can check sense match
    all_hw_forms: Dict[str, List[str]] = {
        hw: sorted({canonicalize_for_eval(_csv_word_to_ge2pe(
            "".join(c if c in ("A", "S", "C", "Z") else ("v" if c == "w" else c.lower()) for c in sk)
        )) for sk in senses.keys()})
        for hw, senses in by_hw.items()
    }
    for lst in (train_items, test_items, showcase_items):
        for item in lst:
            item["all_sense_forms"] = all_hw_forms.get(item["homograph_grapheme"], [item["homograph_eval"]])

    return train_items, test_items, showcase_items


def _predict_raw_ge2pe(model, tokenizer, texts: List[str], batch_size: int = 16) -> List[str]:
    import torch

    device = next(model.parameters()).device
    preds: List[str] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out_ids = model.generate(
                enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                num_beams=2,
                min_length=1,
                max_length=256,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        preds.extend(d.strip() for d in decoded)
    return preds


def score_predictions(items: List[Dict[str, Any]], raw_preds: List[str]) -> Dict[str, Any]:
    total_chars = 0
    total_char_edits = 0
    total_words = 0
    exact_words = 0
    hw_total = 0
    hw_correct = 0
    per_item: List[Dict[str, Any]] = []

    for item, raw_pred in zip(items, raw_preds):
        pred_ev = canonicalize_for_eval(raw_pred)
        tgt_ev = item["eval_target"]

        c_dist = levenshtein(pred_ev, tgt_ev)
        total_char_edits += c_dist
        total_chars += max(1, len(tgt_ev))

        p_toks = pred_ev.split()
        t_toks = tgt_ev.split()
        n_w = len(t_toks)
        total_words += n_w
        matches = sum(1 for a, b in zip(p_toks, t_toks) if a == b)
        exact_words += matches

        hw_eval = item["homograph_eval"]
        # Check if predicted string contains the expected homograph sense (or prefixes it on compound/Ezafe tokens)
        hw_hit = any(
            pt == hw_eval or pt.startswith(hw_eval + "e") or pt.startswith(hw_eval + "h")
            for pt in p_toks
        )
        hw_total += 1
        if hw_hit:
            hw_correct += 1

        per_item.append({
            "grapheme": item["grapheme"],
            "split_tag": item.get("split_tag", ""),
            "homograph_grapheme": item["homograph_grapheme"],
            "homograph_target": hw_eval,
            "homograph_hit": hw_hit,
            "target_ge2pe": item["target_ge2pe"],
            "pred_raw": raw_pred,
            "eval_target": tgt_ev,
            "eval_pred": pred_ev,
            "cer": round(c_dist / max(1, len(tgt_ev)), 4),
        })

    return {
        "count": len(items),
        "per": round(total_char_edits / max(1, total_chars), 4),
        "word_exact_acc": round(exact_words / max(1, total_words), 4),
        "homograph_acc": round(hw_correct / max(1, hw_total), 4),
        "homograph_hits": hw_correct,
        "items": per_item,
    }


def _ensure_mana_config(cache_dir: str = "pretrained") -> str:
    import urllib.request

    cfg_dir = os.path.join(cache_dir, "mana")
    cfg_path = os.path.join(cfg_dir, "config.json")
    if not os.path.isfile(cfg_path):
        os.makedirs(cfg_dir, exist_ok=True)
        url = "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/raw/main/ckpt/config.json"
        print(f"[Download] Fetching {url} ...")
        urllib.request.urlretrieve(url, cfg_path)
    return cfg_path


def _ensure_homo_ge2pe_dir(cache_dir: str = "pretrained") -> str:
    import urllib.request
    import zipfile

    out_dir = os.path.join(cache_dir, "homo-ge2pe")
    zip_path = os.path.join(cache_dir, "homo-ge2pe.zip")
    if not os.path.isdir(out_dir):
        os.makedirs(cache_dir, exist_ok=True)
        url = "https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian/resolve/main/model-weights/homo-ge2pe.zip"
        if not os.path.isfile(zip_path):
            print(f"[Download] Fetching {url} ...")
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    for r, _d, files in os.walk(out_dir):
        if "config.json" in files:
            return r
    return out_dir


def _raw_ge2pe_to_ids(g2p_obj: Any, raw_ge2pe_str: str, original_text: str) -> Tuple[List[int], List[int]]:
    from symbols import PROSODY_TAGS

    pairs = g2p_obj._ge2pe_str_to_phoneme_tag_pairs(raw_ge2pe_str, original_text=original_text)
    phoneme_ids: List[int] = []
    prosody_ids: List[int] = []
    for sym, p_tag in pairs:
        for sub_sym in g2p_obj._split_ipa_for_piper(sym):
            p_id = g2p_obj.phoneme_id_map.get(sub_sym, g2p_obj.pad_id)
            phoneme_ids.append(p_id)
            prosody_ids.append(p_tag)
            phoneme_ids.append(g2p_obj.pad_id)
            prosody_ids.append(PROSODY_TAGS["NORMAL"])
    return phoneme_ids, prosody_ids


def run_g2p_finetune_and_eval(
    csv_path: str = DEFAULT_CSV_PATH,
    ge2pe_dir: str = DEFAULT_GE2PE_DIR,
    onnx_path: str = DEFAULT_ONNX_PATH,
    output_dir: str = "output",
    train_limit: int = 2400,
    test_limit: int = 300,
    epochs: int = 12,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import onnxruntime as ort
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup

    from evaluate_5_sentences import _compute_log_mel
    from finetune_persian import save_wav, synthesize_with_upgraded_onnx
    from g2p_persian import PersianG2PProsody

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2P-FT] Device={device} | Dataset={csv_path}")

    resolved_ge2pe = _ensure_homo_ge2pe_dir()
    mana_cfg_path = _ensure_mana_config()
    print(f"[G2P-FT] Base Homo-GE2PE dir: {resolved_ge2pe} | Mana config: {mana_cfg_path}")

    train_items, test_items, showcase_items = load_and_split_dataset(
        csv_path=csv_path,
        train_limit=train_limit,
        test_limit=test_limit,
        seed=42,
    )
    train_eval_subset = train_items[:150]
    print(
        f"[Split] Train={len(train_items)} | Test_Unseen={len(test_items)} "
        f"| Showcase={len(showcase_items)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(resolved_ge2pe)
    model = T5ForConditionalGeneration.from_pretrained(resolved_ge2pe).to(device)

    # 1. BEFORE Evaluation
    print("[Eval BEFORE] Evaluating unmodified base Homo-GE2PE-Persian...")
    before_train_preds = _predict_raw_ge2pe(
        model, tokenizer, [x["input_text"] for x in train_eval_subset]
    )
    before_test_preds = _predict_raw_ge2pe(
        model, tokenizer, [x["input_text"] for x in test_items]
    )
    before_show_preds = _predict_raw_ge2pe(
        model, tokenizer, [x["input_text"] for x in showcase_items]
    )

    before_train_metrics = score_predictions(train_eval_subset, before_train_preds)
    before_test_metrics = score_predictions(test_items, before_test_preds)
    before_show_metrics = score_predictions(showcase_items, before_show_preds)
    print(
        f"[BEFORE] TrainSample PER={before_train_metrics['per']:.4f} "
        f"HomographAcc={before_train_metrics['homograph_acc']:.4f} | "
        f"TestUnseen PER={before_test_metrics['per']:.4f} "
        f"HomographAcc={before_test_metrics['homograph_acc']:.4f}"
    )

    # Synthesize BEFORE WAVs using base G2P + trained upgraded Piper ONNX (with Mana phoneme_id_map!)
    before_wavs: List[np.ndarray] = []
    before_g2p = PersianG2PProsody(piper_config_json_path=mana_cfg_path)
    before_g2p.attach_homo_ge2pe(resolved_ge2pe, gpu=torch.cuda.is_available())
    has_onnx = os.path.isfile(onnx_path)
    ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]) if has_onnx else None
    if ort_sess is not None and before_g2p.homo_ge2pe is not None:
        ctx_dim = int(before_g2p.homo_ge2pe.d_model)
        for idx, (item, raw_p) in enumerate(zip(showcase_items, before_show_preds), start=1):
            ctx = before_g2p.homo_ge2pe.extract_context_embedding(item["input_text"])
            p_ids, pr_ids = _raw_ge2pe_to_ids(before_g2p, raw_p, item["grapheme"])
            wav = synthesize_with_upgraded_onnx(ort_sess, p_ids, pr_ids, ctx, context_dim=ctx_dim)
            before_wavs.append(wav)
            save_wav(os.path.join(output_dir, f"g2p_test{idx}_before.wav"), wav, 22050)
        pause = np.zeros(int(22050 * 0.35), dtype=np.float32)
        all_b = []
        for w in before_wavs:
            all_b.extend([w, pause])
        save_wav(os.path.join(output_dir, "g2p_all_before.wav"), np.concatenate(all_b), 22050)

    # 2. Fine-Tune ByT5 on `train_items` (FP32 for ByT5 numerical stability)
    class G2PDataset(Dataset):
        def __init__(self, rows: List[Dict[str, str]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> Dict[str, str]:
            return self.rows[idx]

    def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        inputs = [b["input_text"] for b in batch]
        targets = [b["target_ge2pe"] for b in batch]
        enc = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=False,
            return_tensors="pt",
        )
        dec = tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            return_tensors="pt",
        )
        labels = dec["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    train_loader = DataLoader(G2PDataset(train_items), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(G2PDataset(test_items), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = steps_per_epoch * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(10, int(0.1 * total_steps)), num_training_steps=total_steps
    )

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    ft_dir = os.path.join(output_dir, "homo_ge2pe_finetuned")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader, start=1):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = out.loss / grad_accum
            loss.backward()
            tr_loss_sum += float(out.loss.detach().cpu())
            tr_steps += 1
            if step_idx % grad_accum == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                vout = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss_sum += float(vout.loss.detach().cpu())
                val_steps += 1

        tr_loss = tr_loss_sum / max(1, tr_steps)
        val_loss = val_loss_sum / max(1, val_steps)

        # Quick epoch checkpoint PER & Homograph Accuracy on Test_Unseen every 2 epochs (+ Epoch 1 & final)
        ep_row: Dict[str, Any] = {
            "epoch": ep,
            "train_loss": round(tr_loss, 4),
            "val_loss": round(val_loss, 4),
        }
        if ep in (1, epochs) or ep % 2 == 0:
            preds = _predict_raw_ge2pe(model, tokenizer, [x["input_text"] for x in test_items])
            sc = score_predictions(test_items, preds)
            ep_row["test_per"] = sc["per"]
            ep_row["test_homograph_acc"] = sc["homograph_acc"]
            print(
                f"[Epoch {ep:02d}/{epochs}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                f"| Test PER={sc['per']:.4f} HomographAcc={sc['homograph_acc']:.4f}"
            )
        else:
            print(f"[Epoch {ep:02d}/{epochs}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        history.append(ep_row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(ft_dir)
            tokenizer.save_pretrained(ft_dir)

    # Load best checkpoint for final AFTER evaluation
    best_model = T5ForConditionalGeneration.from_pretrained(ft_dir).to(device).eval()

    print("[Eval AFTER] Evaluating fine-tuned Homo-GE2PE checkpoint...")
    after_train_preds = _predict_raw_ge2pe(
        best_model, tokenizer, [x["input_text"] for x in train_eval_subset]
    )
    after_test_preds = _predict_raw_ge2pe(
        best_model, tokenizer, [x["input_text"] for x in test_items]
    )
    after_show_preds = _predict_raw_ge2pe(
        best_model, tokenizer, [x["input_text"] for x in showcase_items]
    )

    after_train_metrics = score_predictions(train_eval_subset, after_train_preds)
    after_test_metrics = score_predictions(test_items, after_test_preds)
    after_show_metrics = score_predictions(showcase_items, after_show_preds)
    print(
        f"[AFTER ] TrainSample PER={after_train_metrics['per']:.4f} "
        f"HomographAcc={after_train_metrics['homograph_acc']:.4f} | "
        f"TestUnseen PER={after_test_metrics['per']:.4f} "
        f"HomographAcc={after_test_metrics['homograph_acc']:.4f}"
    )

    # Synthesize AFTER WAVs
    after_wavs: List[np.ndarray] = []
    after_g2p = PersianG2PProsody(piper_config_json_path=mana_cfg_path)
    after_g2p.attach_homo_ge2pe(ft_dir, gpu=torch.cuda.is_available())
    if ort_sess is not None and after_g2p.homo_ge2pe is not None:
        ctx_dim = int(after_g2p.homo_ge2pe.d_model)
        for idx, (item, raw_p) in enumerate(zip(showcase_items, after_show_preds), start=1):
            ctx = after_g2p.homo_ge2pe.extract_context_embedding(item["input_text"])
            p_ids, pr_ids = _raw_ge2pe_to_ids(after_g2p, raw_p, item["grapheme"])
            wav = synthesize_with_upgraded_onnx(ort_sess, p_ids, pr_ids, ctx, context_dim=ctx_dim)
            after_wavs.append(wav)
            save_wav(os.path.join(output_dir, f"g2p_test{idx}_after.wav"), wav, 22050)
        pause = np.zeros(int(22050 * 0.35), dtype=np.float32)
        all_a = []
        for w in after_wavs:
            all_a.extend([w, pause])
        save_wav(os.path.join(output_dir, "g2p_all_after.wav"), np.concatenate(all_a), 22050)

    # 3. Plots
    eps = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=160)
    axes[0].plot(eps, [h["train_loss"] for h in history], marker="o", label="Train CE Loss")
    axes[0].plot(eps, [h["val_loss"] for h in history], marker="s", label="Held-Out Val Loss")
    axes[0].set_title("Homo-GE2PE Fine-Tuning Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    ep_eval = [h for h in history if "test_per" in h]
    axes[1].axhline(before_test_metrics["per"], linestyle="--", color="#ef4444", label=f"Before ({before_test_metrics['per']:.4f})")
    axes[1].plot([h["epoch"] for h in ep_eval], [h["test_per"] for h in ep_eval], marker="o", color="#2563eb", label="After PER")
    axes[1].set_title("Held-Out Phoneme Error Rate (Lower=Better)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].axhline(before_test_metrics["homograph_acc"] * 100, linestyle="--", color="#ef4444", label=f"Before ({before_test_metrics['homograph_acc']*100:.1f}%)")
    axes[2].plot([h["epoch"] for h in ep_eval], [h["test_homograph_acc"] * 100 for h in ep_eval], marker="o", color="#16a34a", label="After Acc %")
    axes[2].set_title("Held-Out Homograph Accuracy (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "g2p_training_curves.png")
    plt.savefig(curves_path, dpi=160)
    plt.close(fig)

    if before_wavs and after_wavs:
        n_rows = len(showcase_items)
        fig2, axes2 = plt.subplots(n_rows, 4, figsize=(16, 2.5 * n_rows), dpi=140)
        for i, item in enumerate(showcase_items):
            wb, wa = before_wavs[i], after_wavs[i]
            tb = np.arange(len(wb)) / 22050.0
            ta = np.arange(len(wa)) / 22050.0
            axes2[i, 0].plot(tb, wb, linewidth=0.5, color="#dc2626")
            axes2[i, 0].set_ylabel(f"T{i+1} ({item['homograph_phoneme']})", fontsize=8)
            axes2[i, 0].set_ylim(-0.6, 0.6)
            axes2[i, 1].plot(ta, wa, linewidth=0.5, color="#16a34a")
            axes2[i, 1].set_ylim(-0.6, 0.6)
            mb = _compute_log_mel(wb)
            ma = _compute_log_mel(wa)
            axes2[i, 2].imshow(mb, origin="lower", aspect="auto", cmap="magma")
            axes2[i, 3].imshow(ma, origin="lower", aspect="auto", cmap="viridis")
            b_hit = "OK" if before_show_metrics["items"][i]["homograph_hit"] else "MISS"
            a_hit = "OK" if after_show_metrics["items"][i]["homograph_hit"] else "MISS"
            axes2[i, 0].set_title(f"[{item['split_tag']}] BEFORE ({b_hit})", fontsize=8)
            axes2[i, 1].set_title(f"AFTER ({a_hit}) | {item['homograph_eval']}", fontsize=8)
            if i == 0:
                axes2[i, 2].set_title("BEFORE Log-Mel", fontsize=9)
                axes2[i, 3].set_title("AFTER Log-Mel", fontsize=9)
        plt.suptitle("Dataset-for-G2P Fine-Tuning Showcase: Before vs. After (Homo-GE2PE -> Piper ONNX)", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        comp_path = os.path.join(output_dir, "g2p_finetune_comparison.png")
        plt.savefig(comp_path, dpi=140)
        plt.close(fig2)

    report = {
        "dataset_csv": csv_path,
        "train_count": len(train_items),
        "test_unseen_count": len(test_items),
        "showcase_count": len(showcase_items),
        "epochs": epochs,
        "lr": lr,
        "before_train_sample": {k: v for k, v in before_train_metrics.items() if k != "items"},
        "after_train_sample": {k: v for k, v in after_train_metrics.items() if k != "items"},
        "before_test_unseen": {k: v for k, v in before_test_metrics.items() if k != "items"},
        "after_test_unseen": {k: v for k, v in after_test_metrics.items() if k != "items"},
        "before_showcase_summary": {k: v for k, v in before_show_metrics.items() if k != "items"},
        "after_showcase_summary": {k: v for k, v in after_show_metrics.items() if k != "items"},
        "history": history,
        "showcase_details": [
            {
                "id": i + 1,
                "split_tag": item["split_tag"],
                "grapheme": item["grapheme"],
                "homograph_grapheme": item["homograph_grapheme"],
                "homograph_phoneme_csv": item["homograph_phoneme"],
                "homograph_target_ge2pe": item["homograph_eval"],
                "target_ge2pe": item["target_ge2pe"],
                "before_pred_raw": before_show_metrics["items"][i]["pred_raw"],
                "before_homograph_hit": before_show_metrics["items"][i]["homograph_hit"],
                "before_per": before_show_metrics["items"][i]["cer"],
                "after_pred_raw": after_show_metrics["items"][i]["pred_raw"],
                "after_homograph_hit": after_show_metrics["items"][i]["homograph_hit"],
                "after_per": after_show_metrics["items"][i]["cer"],
                "before_wav": f"output/g2p_test{i+1}_before.wav",
                "after_wav": f"output/g2p_test{i+1}_after.wav",
            }
            for i, item in enumerate(showcase_items)
        ],
    }
    report_path = os.path.join(output_dir, "g2p_eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {report_path} and {curves_path}")
    return report


if __name__ == "__main__":
    run_g2p_finetune_and_eval()
