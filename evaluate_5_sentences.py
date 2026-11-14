"""
Evaluate 5 Challenging Persian Benchmark Sentences (Before vs. After Real VITS Fine-Tuning).

BEFORE (Baseline): Official Piper Persian ONNX (`fa_IR-amir-medium.onnx`) + raw `espeak-ng -v fa`.
AFTER  (Upgraded): Our end-to-end fine-tuned `output/persian_vits_upgraded.onnx`
                   (Mana-Persian-Piper + ProsodyEnhancedEmbedding + DurationPredictorAdapter
                   + FlowWithAttention) driven by `Homo-GE2PE-Persian` (phonemes +
                   prosody tags + 1472-dim ByT5 context embeddings).
Includes independent Persian ASR (Whisper) CER/WER verification + 80-bin Log-Mel visual diff.
"""

import glob
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from scipy.signal import resample_poly

from finetune_persian import (
    FILTER_LENGTH,
    HOP_LENGTH,
    N_MELS,
    TARGET_SR,
    WIN_LENGTH,
    ensure_pretrained_assets,
    save_wav,
    synthesize_with_upgraded_onnx,
)
from g2p_persian import PersianG2PProsody
from piper_vits import mel_spectrogram_torch
from symbols import PAD, PROSODY_TAGS

TAG_ID_TO_NAME = {v: k for k, v in PROSODY_TAGS.items()}

FIVE_BENCHMARK_SENTENCES = [
    {
        "id": 1,
        "text": "حتما در کرم مرد بزرگ شکی نیست، ولی کرم کوچک روی برگ درخت بود.",
        "focus": "hatman (tanvin) + homograph karam-e vs kerm-e + Tashdid (shakki)",
        "short_label": "S1: hatman + karam-e vs kerm-e + Tashdid",
    },
    {
        "id": 2,
        "text": "آن مرد جوان که معلم مدرسه بود، حتما دیروز در حادثه مرد.",
        "focus": "hatman + homograph mard-e vs mord + Tashdid+Ezafe (mo'allem-e)",
        "short_label": "S2: hatman + mard-e vs mord + mo'allem-e",
    },
    {
        "id": 3,
        "text": "آیا مهر استاد دانشگاه حتما با مهر و محبت بر برگه امتحان زده شد؟",
        "focus": "hatman + homograph mohr-e vs mehr + Tashdid (mohabbat) + Question",
        "short_label": "S3: hatman + mohr-e vs mehr + Question",
    },
    {
        "id": 4,
        "text": "مدیر جدید در سمت ریاست، حتما با دقت به سمت اتاق جلسات رفت.",
        "focus": "hatman + homograph semat-e vs samt-e + Tashdid (deqqat)",
        "short_label": "S4: hatman + semat-e vs samt-e + deqqat",
    },
    {
        "id": 5,
        "text": "معلم خوب پس از تشکر فراوان، حتما داستان شیر جنگل را با صدای بلند خواند؟",
        "focus": "hatman + Tashdid+Ezafe (mo'allem-e, tashakkor-e) + 2x Ezafe + Question",
        "short_label": "S5: hatman + tashakkor + shir-e jangal + ?",
    },
]


def discover_homo_ge2pe_dir(base_candidates: Optional[List[str]] = None) -> Optional[str]:
    candidates = base_candidates or [
        os.path.join("pretrained", "homo-ge2pe"),
        "homo-ge2pe-repo",
        "/kaggle/working/homo-ge2pe-repo",
        "Homo-GE2PE-Persian",
    ]
    for base in candidates:
        if os.path.isdir(base):
            cfg_matches = glob.glob(os.path.join(base, "**", "config.json"), recursive=True)
            if cfg_matches:
                cfg_matches.sort(key=lambda p: (0 if "ge2pe" in p.lower() else 1, len(p)))
                return os.path.dirname(cfg_matches[0])
    return None


def get_espeak_ipa(text: str) -> str:
    try:
        res = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", "fa", text],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return " ".join(res.stdout.strip().split())
    except Exception as exc:
        return f"<espeak-ng unavailable: {exc}>"


def espeak_baseline_phoneme_ids(text: str, phoneme_id_map: Dict[str, int]) -> List[int]:
    """Baseline Piper phoneme IDs via raw `espeak-ng -v fa --ipa`."""
    res = subprocess.run(
        ["espeak-ng", "-q", "-v", "fa", "--ipa", text],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    ipa_raw = res.stdout.strip().replace("\n", " ")
    ids = [phoneme_id_map.get("^", 1), phoneme_id_map.get("_", 0)]
    for ch in ipa_raw:
        if ch in phoneme_id_map:
            ids.extend([phoneme_id_map[ch], phoneme_id_map.get("_", 0)])
    ids.extend([phoneme_id_map.get("$", 2), phoneme_id_map.get("_", 0)])
    return ids


def synthesize_baseline_piper_onnx(
    ort_sess: ort.InferenceSession,
    phoneme_ids: List[int],
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w: float = 0.8,
) -> np.ndarray:
    inp = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    inp_len = np.array([inp.shape[1]], dtype=np.int64)
    scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)
    audio = ort_sess.run(None, {"input": inp, "input_lengths": inp_len, "scales": scales})[0]
    return np.squeeze(audio).astype(np.float32)


def _compute_log_mel(wav_np: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    if len(wav_np) < WIN_LENGTH:
        wav_np = np.pad(wav_np, (0, WIN_LENGTH - len(wav_np)))
    wav_t = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        mel = mel_spectrogram_torch(
            wav_t, FILTER_LENGTH, N_MELS, sr, HOP_LENGTH, WIN_LENGTH, 0.0, None
        )
    return mel.squeeze(0).numpy()


def _normalize_persian_eval(text: str) -> str:
    t = text.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی").replace("\u200c", " ")
    t = re.sub(r"[ًٌٍَُِّْ،؛؟.,?!\"«»\-]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    n, m = len(seq_a), len(seq_b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def _score_asr(ref_text: str, hyp_text: str) -> Dict[str, Any]:
    ref_n = _normalize_persian_eval(ref_text)
    hyp_n = _normalize_persian_eval(hyp_text)
    ref_chars = list(ref_n.replace(" ", ""))
    hyp_chars = list(hyp_n.replace(" ", ""))
    ref_words = ref_n.split()
    hyp_words = hyp_n.split()
    cer = _edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))
    wer = _edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    return {"hyp": hyp_text.strip(), "cer": round(cer, 4), "wer": round(wer, 4)}


def _load_whisper_asr():
    try:
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=device,
        )
        print(f"[ASR] Loaded openai/whisper-small on device={device}")
        return asr
    except Exception as exc:
        print(f"[ASR] Whisper ASR unavailable ({exc}); skipping ASR metrics.")
        return None


def _transcribe_wav(asr_pipe: Any, wav_np: np.ndarray, sr: int = TARGET_SR) -> str:
    if asr_pipe is None or len(wav_np) == 0:
        return ""
    wav_16k = resample_poly(np.asarray(wav_np, dtype=np.float32), 320, 441).astype(np.float32)
    out = asr_pipe(
        {"array": wav_16k, "sampling_rate": 16000},
        generate_kwargs={"language": "persian", "task": "transcribe"},
    )
    return str(out.get("text", "")).strip()


def plot_five_sentences_comparison(rows: List[Dict], save_path: str, sr: int = TARGET_SR) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(20, 3.2 * n), dpi=160)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, row in enumerate(rows):
        wav_b = row["wav_before"]
        wav_a = row["wav_after"]
        t_b = np.linspace(0, len(wav_b) / sr, num=max(1, len(wav_b)))
        t_a = np.linspace(0, len(wav_a) / sr, num=max(1, len(wav_a)))

        mel_b = _compute_log_mel(wav_b, sr=sr)
        mel_a = _compute_log_mel(wav_a, sr=sr)

        axes[idx, 0].plot(t_b, wav_b, color="#d9534f", linewidth=0.5)
        axes[idx, 0].set_ylim(-1.05, 1.05)
        axes[idx, 0].set_ylabel(f"S{row['id']}", fontsize=10, fontweight="bold")
        if idx == 0:
            axes[idx, 0].set_title("BEFORE: Waveform (espeak-ng + Amir ONNX)", fontsize=10, fontweight="bold")
        axes[idx, 0].grid(alpha=0.25)

        axes[idx, 1].plot(t_a, wav_a, color="#2ca02c", linewidth=0.5)
        axes[idx, 1].set_ylim(-1.05, 1.05)
        if idx == 0:
            axes[idx, 1].set_title("AFTER: Waveform (Homo-GE2PE + Fine-Tuned Upgraded ONNX)", fontsize=10, fontweight="bold")
        axes[idx, 1].grid(alpha=0.25)

        axes[idx, 2].imshow(mel_b, origin="lower", aspect="auto", cmap="magma")
        if idx == 0:
            axes[idx, 2].set_title("BEFORE: 80-Bin Log-Mel", fontsize=10, fontweight="bold")
        axes[idx, 2].set_yticks([])

        axes[idx, 3].imshow(mel_a, origin="lower", aspect="auto", cmap="viridis")
        if idx == 0:
            axes[idx, 3].set_title("AFTER: 80-Bin Log-Mel", fontsize=10, fontweight="bold")
        axes[idx, 3].set_yticks([])

        axes[idx, 0].text(
            0.02, 0.84, f"Before {row['before_duration_sec']:.2f}s",
            transform=axes[idx, 0].transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        axes[idx, 1].text(
            0.02, 0.84, f"After {row['after_duration_sec']:.2f}s | {row['short_label']}",
            transform=axes[idx, 1].transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        if idx == n - 1:
            axes[idx, 0].set_xlabel("Time (s)")
            axes[idx, 1].set_xlabel("Time (s)")
            axes[idx, 2].set_xlabel("Frame")
            axes[idx, 3].set_xlabel("Frame")

    fig.suptitle(
        "Persian Piper VITS End-to-End Evaluation (Baseline Amir vs. Fine-Tuned Upgraded Mana+Homo-GE2PE ONNX)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def main():
    os.makedirs("output", exist_ok=True)
    assets = ensure_pretrained_assets()

    upgraded_onnx_path = "output/persian_vits_upgraded.onnx"
    if not os.path.isfile(upgraded_onnx_path):
        raise FileNotFoundError(
            f"Missing fine-tuned model at {upgraded_onnx_path}. Run `python finetune_persian.py` first."
        )

    baseline_g2p = PersianG2PProsody(piper_config_json_path=assets["amir_cfg"])
    upgraded_g2p = PersianG2PProsody(piper_config_json_path=assets["mana_cfg"])

    ge2pe_dir = discover_homo_ge2pe_dir([assets["ge2pe_dir"]])
    ge2pe_loaded = False
    if ge2pe_dir:
        ge2pe_loaded = upgraded_g2p.attach_homo_ge2pe(
            model_dir=ge2pe_dir, gpu=torch.cuda.is_available()
        )
    context_dim = (
        int(upgraded_g2p.homo_ge2pe.d_model)
        if (ge2pe_loaded and upgraded_g2p.homo_ge2pe)
        else 1472
    )
    print(f"[G2P] Homo-GE2PE loaded={ge2pe_loaded} (dir={ge2pe_dir}, d_model={context_dim})")

    ort_before = ort.InferenceSession(assets["amir_onnx"], providers=["CPUExecutionProvider"])
    ort_after = ort.InferenceSession(upgraded_onnx_path, providers=["CPUExecutionProvider"])

    asr_pipe = _load_whisper_asr()

    report_items: List[Dict[str, Any]] = []
    plot_rows: List[Dict[str, Any]] = []
    all_before_parts: List[np.ndarray] = []
    all_after_parts: List[np.ndarray] = []
    pause_sil = np.zeros(int(0.45 * TARGET_SR), dtype=np.float32)

    for item in FIVE_BENCHMARK_SENTENCES:
        sid = item["id"]
        text = item["text"]
        print(f"\n=== Sentence {sid} ===")
        print(f"Text: {text}")

        before_ipa = get_espeak_ipa(text)
        before_ids = espeak_baseline_phoneme_ids(text, baseline_g2p.phoneme_id_map)
        wav_before = synthesize_baseline_piper_onnx(
            ort_before, before_ids, length_scale=1.0, noise_scale=0.667, noise_w=0.8
        )
        before_wav_path = f"output/sent{sid}_before.wav"
        save_wav(before_wav_path, wav_before, sr=TARGET_SR)

        after_syms, after_ids, after_tags, context_emb, raw_ge2pe = (
            upgraded_g2p.text_to_aligned_streams_with_context(
                text, intersperse_blank=True, prefer_homo_ge2pe=ge2pe_loaded
            )
        )
        unmapped_syms = sorted(
            {
                sym
                for sym, pid in zip(after_syms, after_ids)
                if sym != PAD and pid == upgraded_g2p.pad_id
            }
        )

        is_question = "؟" in text or "?" in text
        wav_after = synthesize_with_upgraded_onnx(
            ort_after,
            after_ids,
            after_tags,
            context_emb,
            context_dim=context_dim,
            length_scale=0.98 if is_question else 1.0,
            noise_scale=0.667,
            noise_w=0.8,
        )
        after_wav_path = f"output/sent{sid}_after.wav"
        save_wav(after_wav_path, wav_after, sr=TARGET_SR)

        all_before_parts.extend([wav_before, pause_sil])
        all_after_parts.extend([wav_after, pause_sil])

        tag_counts = {TAG_ID_TO_NAME[k]: after_tags.count(k) for k in sorted(TAG_ID_TO_NAME)}
        before_dur = len(wav_before) / TARGET_SR
        after_dur = len(wav_after) / TARGET_SR
        before_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(wav_before ** 2)) + 1e-8))
        after_rms_db = float(20.0 * np.log10(np.sqrt(np.mean(wav_after ** 2)) + 1e-8))
        after_ipa_clean = "".join(s for s in after_syms if s not in (PAD, "^", "$"))

        before_asr = _score_asr(text, _transcribe_wav(asr_pipe, wav_before)) if asr_pipe else None
        after_asr = _score_asr(text, _transcribe_wav(asr_pipe, wav_after)) if asr_pipe else None

        print(f"BEFORE espeak IPA: {before_ipa}")
        print(f"AFTER  raw GE2PE : {raw_ge2pe or '<fallback>'}")
        print(f"AFTER  Piper IPA : {after_ipa_clean}")
        print(f"Prosody tags: {tag_counts} | unmapped={unmapped_syms}")
        print(f"Duration: Before={before_dur:.2f}s vs After={after_dur:.2f}s")
        if before_asr and after_asr:
            print(f"ASR BEFORE (CER={before_asr['cer']}, WER={before_asr['wer']}): {before_asr['hyp']}")
            print(f"ASR AFTER  (CER={after_asr['cer']}, WER={after_asr['wer']}): {after_asr['hyp']}")

        report_items.append(
            {
                "id": sid,
                "text": text,
                "focus": item["focus"],
                "before_espeak_ipa": before_ipa,
                "before_phoneme_ids_count": len(before_ids),
                "before_duration_sec": round(before_dur, 3),
                "before_rms_db": round(before_rms_db, 2),
                "before_asr": before_asr,
                "after_engine": "Homo-GE2PE + Fine-Tuned Upgraded Mana ONNX" if ge2pe_loaded else "Fallback + Fine-Tuned Upgraded Mana ONNX",
                "after_raw_ge2pe": raw_ge2pe,
                "after_piper_ipa": after_ipa_clean,
                "after_phoneme_ids_count": len(after_ids),
                "after_unmapped_symbols": unmapped_syms,
                "after_duration_sec": round(after_dur, 3),
                "after_rms_db": round(after_rms_db, 2),
                "after_asr": after_asr,
                "prosody_tag_counts": tag_counts,
                "context_emb_shape": list(context_emb.shape) if context_emb is not None else None,
                "before_wav": before_wav_path,
                "after_wav": after_wav_path,
            }
        )
        plot_rows.append(
            {
                "id": sid,
                "short_label": item["short_label"],
                "wav_before": wav_before,
                "wav_after": wav_after,
                "before_duration_sec": before_dur,
                "after_duration_sec": after_dur,
            }
        )

    save_wav("output/all_5_before.wav", np.concatenate(all_before_parts[:-1]), sr=TARGET_SR)
    save_wav("output/all_5_after.wav", np.concatenate(all_after_parts[:-1]), sr=TARGET_SR)

    comp_png_path = "output/five_sentences_comparison.png"
    plot_five_sentences_comparison(plot_rows, comp_png_path, sr=TARGET_SR)

    avg_before_cer = (
        round(float(np.mean([r["before_asr"]["cer"] for r in report_items])), 4)
        if asr_pipe else None
    )
    avg_after_cer = (
        round(float(np.mean([r["after_asr"]["cer"] for r in report_items])), 4)
        if asr_pipe else None
    )
    summary_payload = {
        "homo_ge2pe_loaded": ge2pe_loaded,
        "upgraded_onnx_path": upgraded_onnx_path,
        "upgraded_onnx_size_mb": round(os.path.getsize(upgraded_onnx_path) / 1e6, 2),
        "sample_rate": TARGET_SR,
        "avg_before_asr_cer": avg_before_cer,
        "avg_after_asr_cer": avg_after_cer,
        "sentences": report_items,
    }
    report_json_path = "output/five_sentences_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved visual comparison: {comp_png_path}")
    print(f"Saved JSON report:       {report_json_path}")
    if asr_pipe:
        print(f"Mean ASR CER: Before={avg_before_cer} vs After={avg_after_cer}")


if __name__ == "__main__":
    main()
