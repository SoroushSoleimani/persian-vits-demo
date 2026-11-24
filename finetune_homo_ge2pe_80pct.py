"""Fine-tune MahtaFetrat/Homo-GE2PE-Persian on 80% Repr.-2 Converted G2P Dataset.

Uses `g2p_dataset_converter.build_80_20_dataset` (4,130 train / 1,056 unseen test / 8 showcase),
evaluates BEFORE vs AFTER on PER, Word Exact Accuracy, and Homograph Sense Accuracy,
and synthesizes end-to-end Piper VITS ONNX comparisons.
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from g2p_dataset_converter import (
    DEFAULT_CSV_PATH,
    build_80_20_dataset,
    canonicalize_repr2_for_eval,
)

DEFAULT_ONNX_PATH = os.path.join("output", "persian_vits_upgraded.onnx")


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


def _predict_raw_ge2pe(model, tokenizer, texts: List[str], batch_size: int = 16) -> List[str]:
    import torch

    device = next(model.parameters()).device
    preds: List[str] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_txt = texts[i : i + batch_size]
            enc = tokenizer(
                batch_txt,
                padding=True,
                add_special_tokens=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
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


def score_predictions(items: List[Dict[str, str]], raw_preds: List[str]) -> Dict[str, Any]:
    total_chars = 0
    total_char_edits = 0
    total_words = 0
    exact_words = 0
    hw_hits = 0
    per_item: List[Dict[str, Any]] = []

    for item, raw_p in zip(items, raw_preds):
        ref = item["eval_target"]
        hyp = canonicalize_repr2_for_eval(raw_p)
        edits = levenshtein(ref, hyp)
        total_char_edits += edits
        total_chars += max(1, len(ref))

        ref_w = ref.split()
        hyp_w = hyp.split()
        total_words += len(ref_w)
        for rw, hw in zip(ref_w, hyp_w):
            if rw == hw:
                exact_words += 1

        hw_eval = item["homograph_eval"]
        hw_hit = hw_eval in hyp_w
        if hw_hit:
            hw_hits += 1

        per_item.append(
            {
                "pred_raw": raw_p,
                "pred_eval": hyp,
                "cer": round(edits / max(1, len(ref)), 4),
                "homograph_hit": hw_hit,
            }
        )

    return {
        "count": len(items),
        "per": round(total_char_edits / max(1, total_chars), 4),
        "word_exact_acc": round(exact_words / max(1, total_words), 4),
        "homograph_acc": round(hw_hits / max(1, len(items)), 4),
        "homograph_hits": hw_hits,
        "items": per_item,
    }


def _ensure_mana_config(cache_dir: str = "pretrained") -> str:
    import urllib.request

    cfg_dir = os.path.join(cache_dir, "mana")
    cfg_path = os.path.join(cfg_dir, "config.json")
    if not os.path.isfile(cfg_path):
        os.makedirs(cfg_dir, exist_ok=True)
        url = "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/raw/main/ckpt/config.json"
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
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    for r, _d, files in os.walk(out_dir):
        if "config.json" in files:
            return r
    return out_dir


def _raw_ge2pe_to_ids(g2p_obj: Any, raw_ge2pe_str: str, original_text: str) -> Tuple[List[int], List[int]]:
    from symbols import PROSODY_TAGS

    ge2pe_text = raw_ge2pe_str.strip()
    if ge2pe_text and ge2pe_text[-1] not in ".!?؟،؛":
        ge2pe_text = ge2pe_text + ("?" if ("؟" in original_text or "?" in original_text) else ".")

    pairs = g2p_obj._ge2pe_str_to_phoneme_tag_pairs(ge2pe_text, original_text=original_text)
    phoneme_ids: List[int] = [g2p_obj.pad_id]
    prosody_ids: List[int] = [PROSODY_TAGS["NORMAL"]]
    for sym, p_tag in pairs:
        for sub_sym in g2p_obj._split_ipa_for_piper(sym):
            p_id = g2p_obj.phoneme_id_map.get(sub_sym, g2p_obj.pad_id)
            phoneme_ids.append(p_id)
            prosody_ids.append(p_tag)
            phoneme_ids.append(g2p_obj.pad_id)
            prosody_ids.append(PROSODY_TAGS["NORMAL"])
    return phoneme_ids, prosody_ids


def _clean_tts_audio(wav: np.ndarray, sr: int = 22050, target_peak: float = 0.85) -> np.ndarray:
    arr = np.asarray(wav, dtype=np.float32).flatten()
    if len(arr) == 0:
        return arr
    active = np.where(np.abs(arr) > 0.008)[0]
    if len(active) > 0:
        pad_n = int(0.06 * sr)
        arr = arr[max(0, int(active[0]) - pad_n) : min(len(arr), int(active[-1]) + pad_n)]
    peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
    if peak > 1e-4:
        arr = arr * (float(target_peak) / peak)
    return np.clip(arr, -1.0, 1.0)


def run_80pct_finetune(
    splits_json_path: str = os.path.join("Dataset for G2P", "pre_splits_80_20.json"),
    onnx_path: str = DEFAULT_ONNX_PATH,
    output_dir: str = "output",
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import onnxruntime as ort
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup

    from finetune_persian import save_wav, synthesize_with_upgraded_onnx
    from g2p_persian import PersianG2PProsody

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2P-80%] Device={device} | Epochs={epochs} | Batch={batch_size}")

    resolved_ge2pe = _ensure_homo_ge2pe_dir()
    mana_cfg_path = _ensure_mana_config()

    if os.path.isfile(splits_json_path):
        with open(splits_json_path, "r", encoding="utf-8") as f:
            sp = json.load(f)
        train_items, test_items, showcase_items = sp["train"], sp["test"], sp["showcase"]
        audit_summary = sp.get("summary", {})
    else:
        train_items, test_items, showcase_items, audit_summary = build_80_20_dataset(DEFAULT_CSV_PATH)

    print(f"[Split] Train={len(train_items)} | TestUnseen={len(test_items)} | Showcase={len(showcase_items)}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_ge2pe)
    model = T5ForConditionalGeneration.from_pretrained(resolved_ge2pe).to(device)

    print("[Eval BEFORE] Evaluating base Homo-GE2PE on 20% unseen test set...")
    before_test_preds = _predict_raw_ge2pe(model, tokenizer, [x["input_text"] for x in test_items])
    before_show_preds = _predict_raw_ge2pe(model, tokenizer, [x["input_text"] for x in showcase_items])
    before_test_metrics = score_predictions(test_items, before_test_preds)
    before_show_metrics = score_predictions(showcase_items, before_show_preds)
    print(
        f"[BEFORE] TestUnseen({len(test_items)}) PER={before_test_metrics['per']:.4f} "
        f"WordAcc={before_test_metrics['word_exact_acc']:.4f} HomographAcc={before_test_metrics['homograph_acc']:.4f}"
    )

    has_onnx = os.path.isfile(onnx_path)
    ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]) if has_onnx else None
    before_g2p = PersianG2PProsody(piper_config_json_path=mana_cfg_path)
    before_g2p.attach_homo_ge2pe(resolved_ge2pe, gpu=False)
    if ort_sess is not None and before_g2p.homo_ge2pe is not None:
        ctx_dim = int(before_g2p.homo_ge2pe.d_model)
        for idx, (item, raw_p) in enumerate(zip(showcase_items, before_show_preds), start=1):
            ctx = before_g2p.homo_ge2pe.extract_context_embedding(item["input_text"])
            p_ids, pr_ids = _raw_ge2pe_to_ids(before_g2p, raw_p, item["grapheme"])
            if hasattr(ort, "set_seed"):
                ort.set_seed(42)
            wav = _clean_tts_audio(
                synthesize_with_upgraded_onnx(
                    ort_sess, p_ids, pr_ids, ctx, context_dim=ctx_dim, length_scale=1.0, noise_scale=0.5, noise_w=0.35
                )
            )
            save_wav(os.path.join(output_dir, f"g2p80_test{idx}_before.wav"), wav, 22050)

    class G2PDataset(Dataset):
        def __init__(self, rows: List[Dict[str, str]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> Dict[str, str]:
            return self.rows[idx]

    def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, Any]:
        enc = tokenizer(
            [b["input_text"] for b in batch],
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tgt = tokenizer(
            [b["target_ge2pe"] for b in batch],
            padding=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"].masked_fill(tgt["attention_mask"].ne(1), -100)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    train_loader = DataLoader(G2PDataset(train_items), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(G2PDataset(test_items), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(10, len(train_loader) // 2), num_training_steps=len(train_loader) * epochs)

    ft_dir = os.path.join(output_dir, "homo_ge2pe_80pct_finetuned")
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        for batch in train_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            tr_loss_sum += float(out.loss.detach().cpu())

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                vout = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss_sum += float(vout.loss.detach().cpu())

        tr_loss = tr_loss_sum / max(1, len(train_loader))
        val_loss = val_loss_sum / max(1, len(val_loader))
        ep_row: Dict[str, Any] = {"epoch": ep, "train_loss": round(tr_loss, 4), "val_loss": round(val_loss, 4)}
        if ep in (1, epochs) or ep % 2 == 0:
            preds = _predict_raw_ge2pe(model, tokenizer, [x["input_text"] for x in test_items])
            sc = score_predictions(test_items, preds)
            ep_row["test_per"] = sc["per"]
            ep_row["test_homograph_acc"] = sc["homograph_acc"]
            print(f"[Epoch {ep:02d}/{epochs}] tr={tr_loss:.4f} val={val_loss:.4f} | Test PER={sc['per']:.4f} HomographAcc={sc['homograph_acc']:.4f}")
        else:
            print(f"[Epoch {ep:02d}/{epochs}] tr={tr_loss:.4f} val={val_loss:.4f}")
        history.append(ep_row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(ft_dir)
            tokenizer.save_pretrained(ft_dir)

    best_model = T5ForConditionalGeneration.from_pretrained(ft_dir).to(device).eval()
    after_test_preds = _predict_raw_ge2pe(best_model, tokenizer, [x["input_text"] for x in test_items])
    after_show_preds = _predict_raw_ge2pe(best_model, tokenizer, [x["input_text"] for x in showcase_items])
    after_test_metrics = score_predictions(test_items, after_test_preds)
    after_show_metrics = score_predictions(showcase_items, after_show_preds)
    print(
        f"[AFTER ] TestUnseen({len(test_items)}) PER={after_test_metrics['per']:.4f} "
        f"WordAcc={after_test_metrics['word_exact_acc']:.4f} HomographAcc={after_test_metrics['homograph_acc']:.4f}"
    )

    after_g2p = PersianG2PProsody(piper_config_json_path=mana_cfg_path)
    after_g2p.attach_homo_ge2pe(ft_dir, gpu=False)
    if ort_sess is not None and after_g2p.homo_ge2pe is not None:
        ctx_dim = int(after_g2p.homo_ge2pe.d_model)
        for idx, (item, raw_p) in enumerate(zip(showcase_items, after_show_preds), start=1):
            ctx = after_g2p.homo_ge2pe.extract_context_embedding(item["input_text"])
            p_ids, pr_ids = _raw_ge2pe_to_ids(after_g2p, raw_p, item["grapheme"])
            if hasattr(ort, "set_seed"):
                ort.set_seed(42)
            wav = _clean_tts_audio(
                synthesize_with_upgraded_onnx(
                    ort_sess, p_ids, pr_ids, ctx, context_dim=ctx_dim, length_scale=1.0, noise_scale=0.5, noise_w=0.35
                )
            )
            save_wav(os.path.join(output_dir, f"g2p80_test{idx}_after.wav"), wav, 22050)

    eps = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=160)
    axes[0].plot(eps, [h["train_loss"] for h in history], marker="o", label="Train Loss")
    axes[0].plot(eps, [h["val_loss"] for h in history], marker="s", label="Val Loss (20% Unseen)")
    axes[0].set_title("80% G2P Fine-Tuning Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    ep_eval = [h for h in history if "test_per" in h]
    axes[1].axhline(before_test_metrics["per"], linestyle="--", color="#ef4444", label=f"Before ({before_test_metrics['per']:.4f})")
    axes[1].plot([h["epoch"] for h in ep_eval], [h["test_per"] for h in ep_eval], marker="o", color="#2563eb", label="After PER")
    axes[1].set_title("20% Unseen PER (Lower=Better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].axhline(before_test_metrics["homograph_acc"] * 100, linestyle="--", color="#ef4444", label=f"Before ({before_test_metrics['homograph_acc']*100:.1f}%)")
    axes[2].plot([h["epoch"] for h in ep_eval], [h["test_homograph_acc"] * 100 for h in ep_eval], marker="o", color="#16a34a", label="After Acc %")
    axes[2].set_title("20% Unseen Homograph Acc (%)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "g2p80_training_curves.png"), dpi=160)
    plt.close(fig)

    report = {
        "train_count": len(train_items),
        "test_unseen_count": len(test_items),
        "epochs": epochs,
        "audit_summary": audit_summary,
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
                "raw_phoneme_repr1": item["raw_phoneme_repr1"],
                "target_ge2pe": item["target_ge2pe"],
                "before_pred_raw": before_show_metrics["items"][i]["pred_raw"],
                "before_homograph_hit": before_show_metrics["items"][i]["homograph_hit"],
                "after_pred_raw": after_show_metrics["items"][i]["pred_raw"],
                "after_homograph_hit": after_show_metrics["items"][i]["homograph_hit"],
                "before_wav": f"output/g2p80_test{i+1}_before.wav",
                "after_wav": f"output/g2p80_test{i+1}_after.wav",
            }
            for i, item in enumerate(showcase_items)
        ],
    }
    with open(os.path.join(output_dir, "g2p80_eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    run_80pct_finetune()
