"""Block 9: Export + Inference Pipeline for Persian VITS + Piper Base.

Connects:
  1. Stage 1 Text Normalization (`persian_normalizer.normalize_persian_text`)
  2. Stage 2-4 G2P + Phoneme IDs + Prosody Tags (`g2p_persian.PersianG2PProsody`)
  3. Stage 5-8 ONNX VITS / HiFi-GAN synthesis (`output/persian_vits_upgraded.onnx`)
  4. Stage 9 Sentence/Clause chunking, `scales` (`noise / length / noise_w`),
     interruptible streaming, silence trimming, and peak normalization.
"""

from dataclasses import dataclass, field
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from persian_normalizer import normalize_persian_text

TARGET_SR = 22050

CHUNK_PAUSE_MS = {
    ".": 240,
    "!": 240,
    "?": 260,
    "؟": 260,
    "...": 320,
    "…": 320,
    ",": 160,
    "،": 160,
    ";": 200,
    "؛": 200,
}


@dataclass
class PipelineChunkResult:
    index: int
    raw_text: str
    normalized_text: str
    pause_after_ms: int
    samples: int
    duration_sec: float
    rms_db: float


@dataclass
class PipelineSynthesisResult:
    audio: np.ndarray
    sample_rate: int
    interrupted: bool
    completed_chunks: int
    total_chunks: int
    elapsed_ms: float
    normalized_full_text: str
    scales: Tuple[float, float, float]
    chunks: List[PipelineChunkResult] = field(default_factory=list)
    has_nan_or_inf: bool = False


def chunk_text_for_tts(text: str) -> List[Tuple[str, int]]:
    """Split normalized Persian text into sentence/clause chunks keeping terminal punctuation."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    parts = re.split(r"(\.\.\.|…|[،؛؟.,!?])", cleaned)
    chunks: List[Tuple[str, int]] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if part in CHUNK_PAUSE_MS:
            term = "؟" if part in ("?", "؟") else ("!" if part == "!" else ".")
            chunk_str = buf.strip()
            if chunk_str:
                chunks.append((chunk_str + term, CHUNK_PAUSE_MS[part]))
            buf = ""
        else:
            buf += part
    if buf.strip():
        tail = buf.strip()
        if tail[-1] not in ".!؟?":
            tail += "."
        chunks.append((tail, 0))
    return chunks


def trim_and_normalize_audio(
    wav: np.ndarray,
    sr: int = TARGET_SR,
    silence_thresh: float = 0.008,
    margin_sec: float = 0.05,
    target_peak: float = 0.85,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """Trim excessive leading/trailing VITS silence, apply edge taper, and normalize peak."""
    arr = np.asarray(wav, dtype=np.float32).flatten()
    if len(arr) == 0:
        return arr

    active = np.where(np.abs(arr) > silence_thresh)[0]
    if len(active) > 0:
        margin_n = int(margin_sec * sr)
        start = max(0, int(active[0]) - margin_n)
        end = min(len(arr), int(active[-1]) + margin_n)
        arr = arr[start:end].copy()

    fade_n = min(int(sr * fade_ms / 1000.0), len(arr) // 4)
    if fade_n > 1:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        arr[:fade_n] *= ramp
        arr[-fade_n:] *= ramp[::-1]

    peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
    if peak > 1e-4:
        arr = arr * (float(target_peak) / peak)
    return np.clip(arr, -1.0, 1.0)


def compute_audio_metrics(wav: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
    """Measure duration, RMS dB, peak, voiced F0 (Hz), and silence ratio."""
    arr = np.asarray(wav, dtype=np.float32).flatten()
    if len(arr) == 0:
        return {
            "duration_sec": 0.0,
            "rms_db": -120.0,
            "peak": 0.0,
            "mean_f0_hz": 0.0,
            "silence_ratio": 1.0,
        }

    duration_sec = len(arr) / float(sr)
    rms = float(np.sqrt(np.mean(arr ** 2)) + 1e-12)
    rms_db = float(20.0 * np.log10(rms))
    peak = float(np.max(np.abs(arr)))

    win = max(1, int(0.04 * sr))
    sil_frames = 0
    total_frames = 0
    for i in range(0, max(1, len(arr) - win + 1), win):
        fr = arr[i : i + win]
        total_frames += 1
        if float(np.sqrt(np.mean(fr ** 2))) < 0.012:
            sil_frames += 1
    silence_ratio = sil_frames / float(max(1, total_frames))

    hop = int(0.015 * sr)
    min_lag = max(1, int(sr / 360.0))
    max_lag = min(win - 1, int(sr / 85.0))
    f0s: List[float] = []
    eth = max(1e-6, rms * 0.35)
    for start in range(0, max(1, len(arr) - win + 1), hop):
        frame = arr[start : start + win]
        if float(np.sqrt(np.mean(frame ** 2))) < eth:
            continue
        fz = frame - np.mean(frame)
        denom = float(np.sum(fz ** 2))
        if denom <= 1e-9:
            continue
        ac = np.correlate(fz, fz, mode="full")[len(fz) - 1 :] / denom
        seg = ac[min_lag : max_lag + 1]
        if len(seg) > 0:
            bi = int(np.argmax(seg))
            if float(seg[bi]) > 0.45:
                f0s.append(float(sr / (min_lag + bi)))

    return {
        "duration_sec": round(duration_sec, 4),
        "rms_db": round(rms_db, 2),
        "peak": round(peak, 4),
        "mean_f0_hz": round(float(np.mean(f0s)), 2) if f0s else 0.0,
        "silence_ratio": round(silence_ratio, 3),
    }


class PersianTTSPipeline:
    """Unified Stage 1-9 Persian TTS Pipeline wrapping Normalization, G2P, Prosody, and ONNX VITS."""

    def __init__(
        self,
        g2p: Any,
        synth_ids_fn: Callable[[List[int], List[int], Optional[Any], Tuple[float, float, float]], np.ndarray],
        sr: int = TARGET_SR,
        prefer_homo_ge2pe: bool = False,
    ):
        self.g2p = g2p
        self.synth_ids_fn = synth_ids_fn
        self.sr = sr
        self.prefer_homo_ge2pe = prefer_homo_ge2pe

    def synthesize(
        self,
        raw_text: str,
        scales: Tuple[float, float, float] = (0.5, 1.0, 0.35),
        interrupt_event: Optional[threading.Event] = None,
    ) -> PipelineSynthesisResult:
        t0 = time.perf_counter()
        normalized = normalize_persian_text(raw_text)
        chunks = chunk_text_for_tts(normalized)
        audio_parts: List[np.ndarray] = []
        chunk_results: List[PipelineChunkResult] = []
        interrupted = False

        for idx, (chunk_txt, pause_ms) in enumerate(chunks):
            if interrupt_event is not None and interrupt_event.is_set():
                interrupted = True
                break

            _, p_ids, pr_ids, ctx_emb, _ = self.g2p.text_to_aligned_streams_with_context(
                chunk_txt,
                intersperse_blank=True,
                prefer_homo_ge2pe=self.prefer_homo_ge2pe,
            )
            raw_wav = self.synth_ids_fn(p_ids, pr_ids, ctx_emb, scales)
            clean_wav = trim_and_normalize_audio(raw_wav, sr=self.sr)

            if interrupt_event is not None and interrupt_event.is_set():
                interrupted = True
                break

            audio_parts.append(clean_wav)
            m = compute_audio_metrics(clean_wav, sr=self.sr)
            chunk_results.append(
                PipelineChunkResult(
                    index=idx,
                    raw_text=chunk_txt,
                    normalized_text=chunk_txt,
                    pause_after_ms=pause_ms,
                    samples=len(clean_wav),
                    duration_sec=m["duration_sec"],
                    rms_db=m["rms_db"],
                )
            )
            if pause_ms > 0 and idx < len(chunks) - 1:
                sil_n = int(self.sr * pause_ms / 1000.0)
                if sil_n > 0:
                    audio_parts.append(np.zeros(sil_n, dtype=np.float32))

        merged = (
            np.concatenate(audio_parts).astype(np.float32)
            if audio_parts
            else np.zeros(0, dtype=np.float32)
        )
        has_bad = bool(np.isnan(merged).any() or np.isinf(merged).any()) if len(merged) else False
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return PipelineSynthesisResult(
            audio=merged,
            sample_rate=self.sr,
            interrupted=interrupted,
            completed_chunks=len(chunk_results),
            total_chunks=len(chunks),
            elapsed_ms=elapsed_ms,
            normalized_full_text=normalized,
            scales=scales,
            chunks=chunk_results,
            has_nan_or_inf=has_bad,
        )
