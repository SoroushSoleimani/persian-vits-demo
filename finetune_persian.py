"""
End-to-End GPU Fine-Tuning Pipeline for Upgraded Persian VITS / Piper TTS.

1. Compiles `piper_vits/monotonic_align` Cython extension if not yet built.
2. Loads the 4.2M-step `MahtaFetrat/Mana-Persian-Piper` VITS Generator (`SynthesizerTrn`)
   and Multi-Period Discriminator (`MultiPeriodDiscriminator`) with `strict=True`.
3. Attaches all 3 zero-initialized Persian upgrades (`UpgradedPersianVits`):
     - `ProsodyEnhancedEmbedding` (5 Prosody Tags + 1472-dim `Homo-GE2PE` ByT5 states)
     - `DurationPredictorAdapter` (ConvNeXt + GLU residual on `StochasticDurationPredictor`)
     - `InvertibleFlowAttentionCoupling` (`FlowWithAttention` around `ResidualCouplingBlock`)
4. Streams verified studio Persian clips (`CER <= 0.05`) from `MahtaFetrat/Mana-TTS`
   parquet shards, resamples anti-aliased 44.1 kHz -> 22.05 kHz, and precomputes
   aligned `(phoneme_ids, prosody_ids, context_emb)` via `Homo-GE2PE-Persian`.
5. Trains Generator + Discriminator end-to-end with VITS GAN, Feature Matching,
   Mel-Spectrogram L1, KL Divergence, and MAS Stochastic Duration losses.
6. Exports the complete fine-tuned model (including all 3 upgrades) to
   `output/persian_vits_upgraded.onnx` and verifies dynamic-shape ONNX inference.
"""

import json
import math
import os
import subprocess
import sys
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

from g2p_persian import PersianG2PProsody
from models import (
    UpgradedPersianVits,
    align_context_embedding,
    export_upgraded_vits_onnx,
)
from piper_vits import (
    build_piper_vits_from_checkpoint,
    commons,
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    mel_spectrogram_torch,
    spec_to_mel_torch,
    spectrogram_torch,
)

TARGET_SR = 22050
FILTER_LENGTH = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80

MANA_CKPT_URL = (
    "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/resolve/main/"
    "ckpt/epoch%3D6012-step%3D4203520.ckpt"
)
MANA_CFG_URL = (
    "https://huggingface.co/MahtaFetrat/Mana-Persian-Piper/raw/main/ckpt/config.json"
)
AMIR_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx"
)
AMIR_CFG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/raw/main/"
    "fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx.json"
)
HOMO_GE2PE_ZIP_URL = (
    "https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian/resolve/main/"
    "model-weights/homo-ge2pe.zip"
)


def save_wav(path: str, audio_np: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write mono float32 waveform [-1, 1] to 16-bit PCM WAV."""
    import wave

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clipped = np.clip(np.asarray(audio_np, dtype=np.float32), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def ensure_monotonic_align_built() -> bool:
    """Compile Cython OpenMP `monotonic_align` extension in-place if needed."""
    import piper_vits.monotonic_align as ma

    if ma._maximum_path_c is not None:
        return True
    setup_path = os.path.join("piper_vits", "monotonic_align", "setup.py")
    res = subprocess.run(
        [sys.executable, setup_path, "build_ext", "--inplace"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
        import importlib

        importlib.reload(ma)
    loaded = ma._maximum_path_c is not None
    print(f"[Build] monotonic_align Cython loaded={loaded} (rc={res.returncode})")
    return loaded


def _download_file(url: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    if os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
        return dst_path
    print(f"[Download] {url} -> {dst_path}")
    subprocess.run(["curl", "-sL", "--fail", "-o", dst_path, url], check=True)
    return dst_path


def ensure_pretrained_assets(cache_dir: str = "pretrained") -> Dict[str, str]:
    """Ensure Mana checkpoint/config, baseline Amir ONNX, and Homo-GE2PE weights exist."""
    mana_ckpt = _download_file(MANA_CKPT_URL, os.path.join(cache_dir, "mana", "mana.ckpt"))
    mana_cfg = _download_file(MANA_CFG_URL, os.path.join(cache_dir, "mana", "config.json"))
    amir_onnx = _download_file(AMIR_ONNX_URL, os.path.join(cache_dir, "fa_IR-amir-medium.onnx"))
    amir_cfg = _download_file(AMIR_CFG_URL, os.path.join(cache_dir, "config.json"))

    ge2pe_dir = os.path.join(cache_dir, "homo-ge2pe")
    cfg_found = any(
        fname == "config.json"
        for _, _, files in os.walk(ge2pe_dir)
        for fname in files
    ) if os.path.isdir(ge2pe_dir) else False
    if not cfg_found:
        zip_path = _download_file(HOMO_GE2PE_ZIP_URL, os.path.join(cache_dir, "homo-ge2pe.zip"))
        os.makedirs(ge2pe_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ge2pe_dir)

    return {
        "mana_ckpt": mana_ckpt,
        "mana_cfg": mana_cfg,
        "amir_onnx": amir_onnx,
        "amir_cfg": amir_cfg,
        "ge2pe_dir": ge2pe_dir,
    }


def _decode_audio_field(raw_audio: Any) -> np.ndarray:
    if isinstance(raw_audio, dict):
        if "array" in raw_audio and raw_audio["array"] is not None:
            return np.asarray(raw_audio["array"], dtype=np.float32)
        if "bytes" in raw_audio and raw_audio["bytes"]:
            import io
            import soundfile as sf

            arr, _ = sf.read(io.BytesIO(raw_audio["bytes"]), dtype="float32")
            return np.asarray(arr, dtype=np.float32)
    arr = np.asarray(raw_audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return arr


def _resample_to_target(wav: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr or len(wav) == 0:
        return wav.astype(np.float32)
    g = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    return resample_poly(wav, up=up, down=down).astype(np.float32)


def load_mana_tts_clips(
    max_samples: int = 800,
    max_shards: int = 3,
    cache_dir: str = "/tmp/mana_tts_shards",
    min_dur: float = 1.0,
    max_dur: float = 8.5,
    max_cer: float = 0.05,
) -> List[Dict[str, Any]]:
    """Download Mana-TTS parquet shards directly and filter studio-quality Persian clips."""
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    clips: List[Dict[str, Any]] = []
    min_samples_len = HOP_LENGTH * 40  # >= 40 STFT frames for safe 32-frame segment slicing

    for shard_idx in range(1, max_shards + 1):
        if len(clips) >= max_samples:
            break
        shard_name = f"dataset_part_{shard_idx:03d}.parquet"
        shard_url = (
            f"https://huggingface.co/datasets/MahtaFetrat/Mana-TTS/resolve/main/"
            f"dataset/{shard_name}"
        )
        local_pq = _download_file(shard_url, os.path.join(cache_dir, shard_name))
        df = pd.read_parquet(local_pq)

        for _, row in df.iterrows():
            dur = float(row.get("duration", 0.0))
            cer = float(row.get("CER", 1.0))
            transcript = str(row.get("transcript", "")).strip()
            if not (min_dur <= dur <= max_dur and cer <= max_cer and len(transcript) >= 5):
                continue

            wav = _decode_audio_field(row["audio"])
            orig_sr = int(float(row.get("samplerate", 44100)))
            wav = _resample_to_target(wav, orig_sr, TARGET_SR)
            if len(wav) < min_samples_len:
                continue

            peak = float(np.max(np.abs(wav)))
            if peak < 1e-4:
                continue
            if peak > 0.98:
                wav = wav * (0.95 / peak)

            # Trim tail so sample length aligns cleanly with hop_length=256
            usable = (len(wav) // HOP_LENGTH) * HOP_LENGTH
            wav = np.ascontiguousarray(wav[:usable], dtype=np.float32)

            clips.append({"transcript": transcript, "duration": dur, "cer": cer, "wav": wav})
            if len(clips) >= max_samples:
                break

    print(
        f"[Data] Loaded {len(clips)} studio Mana-TTS clips "
        f"(CER <= {max_cer}, {min_dur}s <= dur <= {max_dur}s, sr={TARGET_SR}Hz)"
    )
    return clips


def _precompute_ge2pe_features(
    clips: List[Dict[str, Any]],
    g2p: PersianG2PProsody,
    context_dim: int,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """Precompute Homo-GE2PE phonemes, prosody tags, and phoneme-aligned context states."""
    items: List[Dict[str, Any]] = []
    ge2pe = getattr(g2p, "homo_ge2pe", None)

    for start in range(0, len(clips), batch_size):
        chunk = clips[start : start + batch_size]
        texts = [c["transcript"] for c in chunk]

        if ge2pe is not None:
            ge2pe_strs = ge2pe.generate_phonemes(texts, use_rules=True)
            device = next(ge2pe.model.parameters()).device
            enc_in = ge2pe.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc_in_dev = {k: v.to(device) for k, v in enc_in.items()}
            with torch.no_grad():
                enc_hidden = ge2pe.model.get_encoder()(**enc_in_dev).last_hidden_state.detach().cpu()
            attn_masks = enc_in["attention_mask"]
        else:
            ge2pe_strs = [None] * len(chunk)
            enc_hidden = None
            attn_masks = None

        for idx, clip in enumerate(chunk):
            if ge2pe is not None and ge2pe_strs[idx] is not None:
                pairs = g2p._ge2pe_str_to_phoneme_tag_pairs(
                    ge2pe_strs[idx], original_text=clip["transcript"]
                )
                real_len = int(attn_masks[idx].sum().item())
                ctx_raw = enc_hidden[idx : idx + 1, : max(1, real_len), :]
            else:
                pairs = g2p._context_fallback_phoneme_tag_pairs(clip["transcript"])
                ctx_raw = None

            p_ids: List[int] = []
            pros_ids: List[int] = []
            for sym, p_tag in pairs:
                for sub_sym in g2p._split_ipa_for_piper(sym):
                    p_ids.append(g2p.phoneme_id_map.get(sub_sym, g2p.pad_id))
                    pros_ids.append(p_tag)
                    p_ids.append(g2p.pad_id)
                    pros_ids.append(0)

            if len(p_ids) < 6:
                continue

            aligned_ctx = (
                align_context_embedding(ctx_raw, target_len=len(p_ids), context_dim=context_dim)
                .squeeze(0)
                .half()
            )
            wav_t = torch.from_numpy(clip["wav"]).unsqueeze(0)
            spec_t = spectrogram_torch(
                wav_t, FILTER_LENGTH, TARGET_SR, HOP_LENGTH, WIN_LENGTH, center=False
            ).squeeze(0)
            if spec_t.size(1) < 36:
                continue

            items.append(
                {
                    "x": torch.tensor(p_ids, dtype=torch.long),
                    "prosody": torch.tensor(pros_ids, dtype=torch.long),
                    "ctx": aligned_ctx,
                    "spec": spec_t,
                    "wav": wav_t.squeeze(0),
                }
            )

    print(f"[Precompute] Prepared {len(items)} aligned VITS training items (context_dim={context_dim})")
    return items


class ManaVitsDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_vits_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_x = max(b["x"].size(0) for b in batch)
    max_spec = max(b["spec"].size(1) for b in batch)
    max_wav = max(b["wav"].size(0) for b in batch)
    bsz = len(batch)
    n_spec_bins = batch[0]["spec"].size(0)
    ctx_dim = batch[0]["ctx"].size(1)

    x_pad = torch.zeros((bsz, max_x), dtype=torch.long)
    p_pad = torch.zeros((bsz, max_x), dtype=torch.long)
    ctx_pad = torch.zeros((bsz, max_x, ctx_dim), dtype=torch.float32)
    spec_pad = torch.zeros((bsz, n_spec_bins, max_spec), dtype=torch.float32)
    wav_pad = torch.zeros((bsz, 1, max_wav), dtype=torch.float32)
    x_len = torch.zeros((bsz,), dtype=torch.long)
    spec_len = torch.zeros((bsz,), dtype=torch.long)

    for i, b in enumerate(batch):
        tx = b["x"].size(0)
        ts = b["spec"].size(1)
        tw = b["wav"].size(0)
        x_pad[i, :tx] = b["x"]
        p_pad[i, :tx] = b["prosody"]
        ctx_pad[i, :tx, :] = b["ctx"].float()
        spec_pad[i, :, :ts] = b["spec"]
        wav_pad[i, 0, :tw] = b["wav"]
        x_len[i] = tx
        spec_len[i] = ts

    return {
        "x": x_pad,
        "x_len": x_len,
        "prosody": p_pad,
        "ctx": ctx_pad,
        "spec": spec_pad,
        "spec_len": spec_len,
        "wav": wav_pad,
    }


def _build_optimizers(
    up_vits: UpgradedPersianVits,
    net_d: nn.Module,
    upgrade_lr: float = 2e-4,
    backbone_lr: float = 2.5e-5,
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    upgrade_prefixes = (
        "vits.enc_p.emb.prosody_emb",
        "vits.enc_p.emb.context_norm",
        "vits.enc_p.emb.context_proj",
        "vits.dp.adapter_blocks",
        "vits.dp.out_proj",
        "vits.dp.cond",
        "vits.flow.attn_coupling",
    )
    upgrade_params, backbone_params = [], []
    for name, param in up_vits.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(upgrade_prefixes):
            upgrade_params.append(param)
        else:
            backbone_params.append(param)

    opt_g = torch.optim.AdamW(
        [
            {"params": upgrade_params, "lr": upgrade_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ],
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=1e-4,
    )
    opt_d = torch.optim.AdamW(
        net_d.parameters(),
        lr=backbone_lr,
        betas=(0.8, 0.99),
        eps=1e-9,
        weight_decay=1e-4,
    )
    return opt_g, opt_d


def plot_training_curves(history: List[Dict[str, float]], out_png: str) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)

    axes[0].plot(epochs, [h["loss_mel"] for h in history], label="Mel L1 (weighted)", color="#0d6efd")
    axes[0].plot(epochs, [h["loss_kl"] for h in history], label="KL Div", color="#6f42c1")
    axes[0].plot(epochs, [h["loss_dur"] for h in history], label="Dur (SDP+Adapter)", color="#198754")
    axes[0].set_title("Acoustic, KL & Duration Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [h["loss_gen"] for h in history], label="Gen Adv", color="#dc3545")
    axes[1].plot(epochs, [h["loss_fm"] for h in history], label="Feat Match", color="#fd7e14")
    axes[1].plot(epochs, [h["loss_disc"] for h in history], label="MPD Disc", color="#20c997")
    axes[1].set_title("GAN & Feature Matching Losses")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, [h["norm_prosody"] for h in history], label="Prosody Emb")
    axes[2].plot(epochs, [h["norm_context_proj"] for h in history], label="ByT5 Context Proj")
    axes[2].plot(epochs, [h["norm_dp_adapter"] for h in history], label="Duration Adapter")
    axes[2].plot(epochs, [h["norm_flow_attn"] for h in history], label="Flow Attn")
    axes[2].set_title("Zero-Init Upgrade Module Norms")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.suptitle("Persian Piper VITS End-to-End Fine-Tuning on Mana-TTS", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def synthesize_with_upgraded_onnx(
    ort_sess: ort.InferenceSession,
    phoneme_ids: List[int],
    prosody_ids: List[int],
    context_emb: Optional[torch.Tensor],
    context_dim: int = 1472,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w: float = 0.8,
) -> np.ndarray:
    """Run ONNX Runtime inference on exported `persian_vits_upgraded.onnx`."""
    inp = np.expand_dims(np.asarray(phoneme_ids, dtype=np.int64), 0)
    inp_len = np.array([inp.shape[1]], dtype=np.int64)
    pros = np.expand_dims(np.asarray(prosody_ids, dtype=np.int64), 0)
    aligned_ctx = align_context_embedding(context_emb, inp.shape[1], context_dim=context_dim).numpy()
    scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

    audio = ort_sess.run(
        None,
        {
            "input": inp,
            "input_lengths": inp_len,
            "prosody_ids": pros,
            "context_emb": aligned_ctx,
            "scales": scales,
        },
    )[0]
    return np.squeeze(audio).astype(np.float32)


def run_finetuning(
    max_samples: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: int = 8,
    output_dir: str = "output",
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    ensure_monotonic_align_built()
    assets = ensure_pretrained_assets()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    if max_samples is None:
        max_samples = 700 if use_gpu else 16
    if epochs is None:
        epochs = 16 if use_gpu else 2

    print(f"[Init] Device={device} | max_samples={max_samples} | epochs={epochs} | batch_size={batch_size}")

    g2p = PersianG2PProsody(piper_config_json_path=assets["mana_cfg"])
    ge2pe_loaded = g2p.attach_homo_ge2pe(model_dir=assets["ge2pe_dir"], gpu=use_gpu)
    context_dim = int(g2p.homo_ge2pe.d_model) if (ge2pe_loaded and g2p.homo_ge2pe) else 1472
    print(f"[G2P] Homo-GE2PE loaded={ge2pe_loaded} | d_model={context_dim}")

    clips = load_mana_tts_clips(max_samples=max_samples)
    items = _precompute_ge2pe_features(clips, g2p, context_dim=context_dim, batch_size=16)

    # Free ByT5 VRAM before loading SynthesizerTrn + MPD onto GPU
    if ge2pe_loaded and g2p.homo_ge2pe is not None and g2p.homo_ge2pe.model is not None:
        g2p.homo_ge2pe.model.cpu()
        if use_gpu:
            torch.cuda.empty_cache()

    model_g, model_d, hp = build_piper_vits_from_checkpoint(
        assets["mana_ckpt"], device="cpu", strict=True
    )
    segment_size = int(model_g.segment_size)
    c_mel = float(hp.get("c_mel", 45.0))

    # Step-0 mathematical invariant check against frozen Mana SynthesizerTrn
    probe_x = items[0]["x"][:32].unsqueeze(0)
    probe_x_len = torch.LongTensor([probe_x.size(1)])
    probe_p = items[0]["prosody"][:32].unsqueeze(0)
    probe_ctx = items[0]["ctx"][:32].unsqueeze(0).float()

    model_g.eval()
    with torch.no_grad():
        torch.manual_seed(1234)
        base_wav, _, _, _ = model_g.infer(
            probe_x, probe_x_len, noise_scale=0.0, length_scale=1.0, noise_scale_w=0.0
        )

    up_vits = UpgradedPersianVits(model_g, context_dim=context_dim).eval()
    with torch.no_grad():
        torch.manual_seed(1234)
        up_wav, _, _, _ = up_vits.infer(
            probe_x,
            probe_x_len,
            prosody_ids=probe_p,
            context_emb=probe_ctx,
            noise_scale=0.0,
            length_scale=1.0,
            noise_scale_w=0.0,
        )
    step0_diff = float((up_wav - base_wav).abs().max().item())
    print(f"[Invariant] Real Mana-Persian-Piper Step-0 max abs diff={step0_diff:.8f}")

    up_vits = up_vits.to(device)
    model_d = model_d.to(device)
    opt_g, opt_d = _build_optimizers(up_vits, model_d)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.96)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.96)

    loader = DataLoader(
        ManaVitsDataset(items),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_vits_batch,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        up_vits.train()
        model_d.train()
        totals = {
            "loss_disc": 0.0,
            "loss_gen": 0.0,
            "loss_fm": 0.0,
            "loss_mel": 0.0,
            "loss_kl": 0.0,
            "loss_dur": 0.0,
            "steps": 0,
        }

        for batch in loader:
            x = batch["x"].to(device)
            x_len = batch["x_len"].to(device)
            prosody = batch["prosody"].to(device)
            ctx = batch["ctx"].to(device)
            spec = batch["spec"].to(device)
            spec_len = batch["spec_len"].to(device)
            wav = batch["wav"].to(device)

            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
                up_vits(
                    x, x_len, spec, spec_len, prosody_ids=prosody, context_emb=ctx
                )
            )

            mel = spec_to_mel_torch(spec, FILTER_LENGTH, N_MELS, TARGET_SR, 0.0, None)
            y_mel = commons.slice_segments(mel, ids_slice, segment_size)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), FILTER_LENGTH, N_MELS, TARGET_SR, HOP_LENGTH, WIN_LENGTH, 0.0, None
            )
            min_mel_t = min(y_mel.size(2), y_hat_mel.size(2))
            y_real_slice = commons.slice_segments(wav, ids_slice * HOP_LENGTH, segment_size * HOP_LENGTH)

            # 1. Discriminator step
            y_d_hat_r, y_d_hat_g, _, _ = model_d(y_real_slice, y_hat.detach())
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            opt_d.zero_grad()
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(model_d.parameters(), 5.0)
            opt_d.step()

            # 2. Generator step
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = model_d(y_real_slice, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel[:, :, :min_mel_t], y_hat_mel[:, :, :min_mel_t]) * c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(up_vits.parameters(), 5.0)
            opt_g.step()

            totals["loss_disc"] += float(loss_disc.item())
            totals["loss_gen"] += float(loss_gen.item())
            totals["loss_fm"] += float(loss_fm.item())
            totals["loss_mel"] += float(loss_mel.item())
            totals["loss_kl"] += float(loss_kl.item())
            totals["loss_dur"] += float(loss_dur.item())
            totals["steps"] += 1

        sched_g.step()
        sched_d.step()

        n_steps = max(1, totals["steps"])
        row = {
            "epoch": epoch,
            "loss_disc": round(totals["loss_disc"] / n_steps, 4),
            "loss_gen": round(totals["loss_gen"] / n_steps, 4),
            "loss_fm": round(totals["loss_fm"] / n_steps, 4),
            "loss_mel": round(totals["loss_mel"] / n_steps, 4),
            "loss_kl": round(totals["loss_kl"] / n_steps, 4),
            "loss_dur": round(totals["loss_dur"] / n_steps, 4),
            "norm_prosody": round(float(up_vits.vits.enc_p.emb.prosody_emb.weight.norm().item()), 5),
            "norm_context_proj": round(float(up_vits.vits.enc_p.emb.context_proj.weight.norm().item()), 5),
            "norm_dp_adapter": round(float(up_vits.vits.dp.out_proj.weight.norm().item()), 5),
            "norm_flow_attn": round(float(up_vits.vits.flow.attn_coupling.out_proj.weight.norm().item()), 5),
        }
        history.append(row)
        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"Mel={row['loss_mel']:.3f} KL={row['loss_kl']:.3f} Dur={row['loss_dur']:.3f} "
            f"Gen={row['loss_gen']:.3f} FM={row['loss_fm']:.3f} Disc={row['loss_disc']:.3f} | "
            f"UpNorms(P={row['norm_prosody']:.4f}, Ctx={row['norm_context_proj']:.4f}, "
            f"DP={row['norm_dp_adapter']:.4f}, Flow={row['norm_flow_attn']:.4f})"
        )

    # Save PyTorch checkpoint & export real upgraded ONNX
    up_vits.eval()
    pt_path = os.path.join(output_dir, "persian_vits_upgraded.pt")
    torch.save(
        {
            "model_g": up_vits.state_dict(),
            "model_d": model_d.state_dict(),
            "context_dim": context_dim,
            "hparams": hp,
            "history": history,
            "step0_diff": step0_diff,
        },
        pt_path,
    )

    onnx_path = os.path.join(output_dir, "persian_vits_upgraded.onnx")
    export_upgraded_vits_onnx(up_vits, onnx_path, opset_version=15)
    onnx_mb = round(os.path.getsize(onnx_path) / 1e6, 2)

    # Verify ONNX Runtime execution on saved model
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_wav = synthesize_with_upgraded_onnx(
        sess,
        items[0]["x"].tolist(),
        items[0]["prosody"].tolist(),
        items[0]["ctx"].unsqueeze(0).float(),
        context_dim=context_dim,
    )
    print(f"[ONNX] Exported {onnx_path} ({onnx_mb} MB) | verified output samples={len(test_wav)}")

    curves_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(history, curves_path)

    if clips:
        save_wav(os.path.join(output_dir, "mana_tts_reference.wav"), clips[0]["wav"], TARGET_SR)

    summary = {
        "device": str(device),
        "num_clips_loaded": len(clips),
        "num_training_items": len(items),
        "epochs": epochs,
        "batch_size": batch_size,
        "context_dim": context_dim,
        "homo_ge2pe_loaded": ge2pe_loaded,
        "step0_diff": step0_diff,
        "onnx_path": onnx_path,
        "onnx_size_mb": onnx_mb,
        "pt_path": pt_path,
        "curves_path": curves_path,
        "history": history,
    }
    hist_path = os.path.join(output_dir, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Done] Saved training history to {hist_path}")
    return summary


if __name__ == "__main__":
    run_finetuning()
