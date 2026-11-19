"""Interruptible Persian TTS Synthesis & Word-Stress Sensitivity Utilities.

Provides:
  - `InterruptibleTTSSynthesizer`: Clause-level chunked synthesis supporting real-time
    `threading.Event` barge-in / cancellation, boundary fade smoothing, and prosodic
    interruption pauses (`,`, `;`, `...`, `-`).
  - `stress_word_to_streams`: Controlled lexical stress positioning (`auto`,
    `first_vowel`, `second_vowel`, `none`) emitting Piper IPA + `STRESS` tags.
  - `analyze_acoustic_prosody`: Pure-NumPy F0 (pitch Hz), RMS dB, and duration metrics.
"""

from dataclasses import dataclass, field
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from symbols import BOS, EOS, PAD, PROSODY_TAGS

TARGET_SR = 22050

PAUSE_MS_BY_PUNCT = {
    ",": 220,
    "،": 220,
    ";": 280,
    "؛": 280,
    "...": 450,
    "…": 450,
    "—": 420,
    "--": 420,
    ".": 380,
    "!": 380,
    "?": 400,
    "؟": 400,
}

VOWEL_IPA_CHARS = set("aɑeɛiouæɒː")


@dataclass
class ChunkMetric:
    index: int
    text: str
    pause_after_ms: int
    samples: int
    duration_ms: float
    rms_db: float


@dataclass
class SynthesisResult:
    audio: np.ndarray
    interrupted: bool
    completed_chunks: int
    total_chunks: int
    elapsed_ms: float
    chunk_metrics: List[ChunkMetric] = field(default_factory=list)
    max_abs_amplitude: float = 0.0
    has_nan_or_inf: bool = False


def split_clauses_with_pauses(text: str) -> List[Tuple[str, int]]:
    """Split Persian utterance into speakable clauses paired with trailing pause (ms)."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    parts = re.split(r"(\.\.\.|…|—|--|[،؛؟.,!?])", cleaned)
    clauses: List[Tuple[str, int]] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if part in PAUSE_MS_BY_PUNCT:
            clause_txt = (buf + part).strip() if part in ("؟", "?", "!") else buf.strip()
            if clause_txt:
                clauses.append((clause_txt, PAUSE_MS_BY_PUNCT[part]))
            buf = ""
        else:
            buf += part

    if buf.strip():
        clauses.append((buf.strip(), 0))
    return clauses


def _apply_edge_fade(wav: np.ndarray, sr: int = TARGET_SR, fade_ms: float = 5.0) -> np.ndarray:
    """Apply short cosine/linear edge fade to prevent click transients on interrupt/join."""
    out = np.asarray(wav, dtype=np.float32).copy()
    fade_n = min(int(sr * fade_ms / 1000.0), len(out) // 4)
    if fade_n > 1:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        out[:fade_n] *= ramp
        out[-fade_n:] *= ramp[::-1]
    return np.clip(out, -1.0, 1.0)


def analyze_acoustic_prosody(wav: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
    """Compute duration, RMS energy (dB), peak, and voiced fundamental frequency (F0 Hz)."""
    arr = np.asarray(wav, dtype=np.float32).flatten()
    if len(arr) == 0:
        return {"duration_sec": 0.0, "rms_db": -120.0, "peak": 0.0, "mean_f0_hz": 0.0, "f0_std_hz": 0.0}

    duration_sec = len(arr) / float(sr)
    rms = float(np.sqrt(np.mean(arr ** 2)) + 1e-12)
    rms_db = float(20.0 * np.log10(rms))
    peak = float(np.max(np.abs(arr)))

    # Short-window autocorrelation F0 estimator over voiced frames (85-360 Hz)
    frame_len = int(0.04 * sr)
    hop_len = int(0.015 * sr)
    min_lag = max(1, int(sr / 360.0))
    max_lag = min(frame_len - 1, int(sr / 85.0))
    f0s: List[float] = []
    energy_thresh = max(1e-6, rms * 0.35)

    for start in range(0, max(1, len(arr) - frame_len + 1), hop_len):
        frame = arr[start : start + frame_len]
        if float(np.sqrt(np.mean(frame ** 2))) < energy_thresh:
            continue
        frame_z = frame - np.mean(frame)
        denom = float(np.sum(frame_z ** 2))
        if denom <= 1e-9:
            continue
        ac = np.correlate(frame_z, frame_z, mode="full")[len(frame_z) - 1 :] / denom
        seg = ac[min_lag : max_lag + 1]
        if len(seg) == 0:
            continue
        best_idx = int(np.argmax(seg))
        if float(seg[best_idx]) > 0.45:
            lag = min_lag + best_idx
            f0s.append(float(sr / lag))

    return {
        "duration_sec": round(duration_sec, 4),
        "rms_db": round(rms_db, 2),
        "peak": round(peak, 4),
        "mean_f0_hz": round(float(np.mean(f0s)), 2) if f0s else 0.0,
        "f0_std_hz": round(float(np.std(f0s)), 2) if f0s else 0.0,
    }


def stress_word_to_streams(
    g2p: Any,
    ge2pe_token: str,
    stress_mode: str = "auto",
) -> Tuple[List[str], List[int], List[int]]:
    """Build (phonemes_out, phoneme_ids, prosody_ids) for one word with controlled stress.

    stress_mode:
      - 'auto': Use `PersianG2PProsody._ge2pe_word_to_pairs` natural stress placement.
      - 'first_vowel': Place `'ˈ'` + STRESS tag before the 1st vowel.
      - 'second_vowel': Place `'ˈ'` + STRESS tag before the 2nd vowel (if >=2 vowels).
      - 'none': Remove `'ˈ'` and downgrade STRESS tags to NORMAL.
    """
    if stress_mode not in ("auto", "first_vowel", "second_vowel", "none"):
        raise ValueError(f"Unsupported stress_mode: {stress_mode!r}")

    base_pairs = g2p._ge2pe_word_to_pairs(ge2pe_token, is_final_question_word=False)

    if stress_mode != "auto":
        # Strip existing explicit stress mark & reset STRESS -> NORMAL first
        clean_pairs: List[Tuple[str, int]] = []
        for sym, tag in base_pairs:
            if sym == "ˈ":
                continue
            new_tag = PROSODY_TAGS["NORMAL"] if tag == PROSODY_TAGS["STRESS"] else tag
            clean_pairs.append((sym, new_tag))

        if stress_mode in ("first_vowel", "second_vowel"):
            vowel_positions = [
                idx
                for idx, (sym, tag) in enumerate(clean_pairs)
                if any(ch in VOWEL_IPA_CHARS for ch in sym) and tag != PROSODY_TAGS["EZAFE"]
            ]
            if vowel_positions:
                target_pos = (
                    vowel_positions[0]
                    if stress_mode == "first_vowel" or len(vowel_positions) < 2
                    else vowel_positions[1]
                )
                sym_at, tag_at = clean_pairs[target_pos]
                clean_pairs[target_pos] = (
                    sym_at,
                    PROSODY_TAGS["STRESS"] if tag_at == PROSODY_TAGS["NORMAL"] else tag_at,
                )
                clean_pairs.insert(target_pos, ("ˈ", PROSODY_TAGS["STRESS"]))
        base_pairs = clean_pairs

    full_pairs = [(BOS, PROSODY_TAGS["NORMAL"])] + base_pairs + [(EOS, PROSODY_TAGS["NORMAL"])]
    syms_out: List[str] = [PAD]
    ids_out: List[int] = [g2p.pad_id]
    tags_out: List[int] = [PROSODY_TAGS["NORMAL"]]
    for sym, tag in full_pairs:
        for sub in g2p._split_ipa_for_piper(sym):
            syms_out.append(sub)
            ids_out.append(g2p.phoneme_id_map.get(sub, g2p.pad_id))
            tags_out.append(tag)
            syms_out.append(PAD)
            ids_out.append(g2p.pad_id)
            tags_out.append(PROSODY_TAGS["NORMAL"])
    return syms_out, ids_out, tags_out


class InterruptibleTTSSynthesizer:
    """Clause-streaming Persian TTS controller supporting clean mid-utterance interruption."""

    def __init__(
        self,
        synth_clause_fn: Callable[[str], np.ndarray],
        sr: int = TARGET_SR,
    ):
        self.synth_clause_fn = synth_clause_fn
        self.sr = sr

    def synthesize_stream(
        self,
        text: str,
        interrupt_event: Optional[threading.Event] = None,
        on_chunk: Optional[Callable[[int, int, str, np.ndarray], None]] = None,
    ) -> SynthesisResult:
        t0 = time.perf_counter()
        clauses = split_clauses_with_pauses(text)
        total = len(clauses)
        audio_parts: List[np.ndarray] = []
        metrics: List[ChunkMetric] = []
        interrupted = False

        for idx, (clause_text, pause_ms) in enumerate(clauses):
            if interrupt_event is not None and interrupt_event.is_set():
                interrupted = True
                break

            raw_chunk = _apply_edge_fade(self.synth_clause_fn(clause_text), sr=self.sr)

            if interrupt_event is not None and interrupt_event.is_set():
                interrupted = True
                break

            audio_parts.append(raw_chunk)
            rms = float(np.sqrt(np.mean(raw_chunk ** 2)) + 1e-12) if len(raw_chunk) else 1e-12
            metrics.append(
                ChunkMetric(
                    index=idx,
                    text=clause_text,
                    pause_after_ms=pause_ms,
                    samples=len(raw_chunk),
                    duration_ms=round(1000.0 * len(raw_chunk) / self.sr, 2),
                    rms_db=round(float(20.0 * np.log10(rms)), 2),
                )
            )

            if on_chunk is not None:
                on_chunk(idx, total, clause_text, raw_chunk)

            if interrupt_event is not None and interrupt_event.is_set():
                interrupted = True
                break

            if pause_ms > 0 and idx < total - 1:
                sil_samples = int(self.sr * pause_ms / 1000.0)
                if sil_samples > 0:
                    audio_parts.append(np.zeros(sil_samples, dtype=np.float32))

        merged = (
            np.concatenate(audio_parts).astype(np.float32)
            if audio_parts
            else np.zeros(0, dtype=np.float32)
        )
        has_bad = bool(np.isnan(merged).any() or np.isinf(merged).any()) if len(merged) else False
        max_amp = float(np.max(np.abs(merged))) if len(merged) else 0.0
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return SynthesisResult(
            audio=merged,
            interrupted=interrupted,
            completed_chunks=len(metrics),
            total_chunks=total,
            elapsed_ms=elapsed_ms,
            chunk_metrics=metrics,
            max_abs_amplitude=round(max_amp, 5),
            has_nan_or_inf=has_bad,
        )
