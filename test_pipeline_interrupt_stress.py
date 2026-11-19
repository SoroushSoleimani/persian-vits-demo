"""Pipeline Test Suite: Interrupt Handling & Lexical Word Stress in Persian TTS.

Covers:
  1. Lexical stress placement (`'ˈ'` + `PROSODY_TAGS["STRESS"]`) across adverbs,
     `mi-`/`nemi-` verbs, Ezafe nouns, clitics, geminated words, and stress shifts.
  2. Clause-level streaming interruption (`threading.Event` barge-in, pre-cancel,
     clean recovery across subsequent calls, zero edge clicks, and prosodic pauses).
  3. Real ONNX synthesis verification (`output/persian_vits_upgraded.onnx`) when
     `onnxruntime` + model weights are present, saving WAVs, JSON report, and plots.
"""

import json
import os
import threading
import unittest
from typing import Any, Dict, List

import numpy as np

from g2p_persian import PersianG2PProsody
from symbols import PROSODY_TAGS
from test_pipeline import MANA_ACTIVE_PHONEME_MAP
from tts_interrupt_stress import (
    TARGET_SR,
    InterruptibleTTSSynthesizer,
    analyze_acoustic_prosody,
    split_clauses_with_pauses,
    stress_word_to_streams,
)

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def _save_wav_pcm16(path: str, wav: np.ndarray, sr: int = TARGET_SR) -> None:
    import wave

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clipped = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class TestStressInWords(unittest.TestCase):
    def setUp(self):
        self.g2p = PersianG2PProsody()
        self.g2p.phoneme_id_map = dict(MANA_ACTIVE_PHONEME_MAP)

    def _ipa_str(self, ge2pe_tok: str, mode: str = "auto") -> str:
        syms, ids, tags = stress_word_to_streams(self.g2p, ge2pe_tok, stress_mode=mode)
        self.assertEqual(len(syms), len(ids))
        self.assertEqual(len(ids), len(tags))
        for s in syms:
            if s not in ("^", "$", "_"):
                self.assertIn(s, self.g2p.phoneme_id_map, f"Unmapped {s!r}")
        return "".join(s for s in syms if s not in ("^", "$", "_"))

    def test_adverb_and_verb_prefix_stress(self):
        self.assertEqual(self._ipa_str("h/tm/n"), "hˈatman")
        self.assertEqual(self._ipa_str("hala"), "hˈɑlɑ")
        self.assertEqual(self._ipa_str("m/s/l/n"), "masˈalan".replace("asˈa", "ˈasa"))
        self.assertEqual(self._ipa_str("mir/v/m"), "mˈiravam")
        self.assertEqual(self._ipa_str("nemixan/d"), "nˈemixɑnad")

    def test_noun_ezafe_and_clitic_stress(self):
        # Final-syllable stem stress preserved; trailing Ezafe '-e' stays unstressed (tag=EZAFE)
        self.assertEqual(self._ipa_str("ketab"), "ketˈɑb")
        self.assertEqual(self._ipa_str("ketabe1"), "ketˈɑbe")
        syms, _, tags = stress_word_to_streams(self.g2p, "ketabe1", stress_mode="auto")
        self.assertIn(PROSODY_TAGS["EZAFE"], tags)
        self.assertIn(PROSODY_TAGS["STRESS"], tags)
        # Clitics must carry zero stress mark and zero STRESS tag
        for clitic in ("o", "be", "/z", "d/r", "ra", "ke", "ba", "ta", "ya"):
            ipa = self._ipa_str(clitic)
            self.assertNotIn("ˈ", ipa, f"Clitic {clitic!r} unexpectedly stressed: {ipa}")

    def test_controlled_word_stress_shifts(self):
        # Same word with initial, second-syllable, and stripped stress modes
        self.assertEqual(self._ipa_str("h/tm/n", "first_vowel"), "hˈatman")
        self.assertEqual(self._ipa_str("h/tm/n", "second_vowel"), "hatmˈan")
        self.assertEqual(self._ipa_str("h/tm/n", "none"), "hatman")

        self.assertEqual(self._ipa_str("hala", "first_vowel"), "hˈɑlɑ")
        self.assertEqual(self._ipa_str("hala", "second_vowel"), "hɑlˈɑ")
        self.assertEqual(self._ipa_str("hala", "none"), "hɑlɑ")

        _, _, tags_none = stress_word_to_streams(self.g2p, "hala", stress_mode="none")
        self.assertNotIn(PROSODY_TAGS["STRESS"], tags_none)


class TestInterruptInTTS(unittest.TestCase):
    @staticmethod
    def _dummy_tone_clause(clause_text: str) -> np.ndarray:
        n_samples = int(TARGET_SR * max(0.18, min(0.6, 0.03 * len(clause_text))))
        t = np.arange(n_samples, dtype=np.float32) / TARGET_SR
        return (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    def test_clause_splitting_and_pause_durations(self):
        text = "مدیر جدید، حتما با دقت... به سمت اتاق رفت؟"
        clauses = split_clauses_with_pauses(text)
        self.assertEqual(len(clauses), 3)
        self.assertLess(clauses[0][1], clauses[1][1])  # ellipsis pause > comma pause
        self.assertEqual(clauses[1][1], 450)
        self.assertEqual(clauses[2][1], 400)

    def test_uninterrupted_and_barge_in_recovery(self):
        synth = InterruptibleTTSSynthesizer(self._dummy_tone_clause)
        text = "سلام، حالتان چطور است؟ حتما خبرهای خوبی دارم."

        # 1. Full uninterrupted run
        res_full = synth.synthesize_stream(text)
        self.assertFalse(res_full.interrupted)
        self.assertEqual(res_full.completed_chunks, res_full.total_chunks)
        self.assertGreater(len(res_full.audio), 0)
        self.assertFalse(res_full.has_nan_or_inf)
        self.assertLessEqual(res_full.max_abs_amplitude, 1.0)

        # 2. Mid-synthesis interrupt right after chunk 0 completes
        stop_ev = threading.Event()

        def on_chunk(idx: int, _total: int, _txt: str, chunk: np.ndarray):
            self.assertGreater(len(chunk), 0)
            if idx == 0:
                stop_ev.set()

        res_int = synth.synthesize_stream(text, interrupt_event=stop_ev, on_chunk=on_chunk)
        self.assertTrue(res_int.interrupted)
        self.assertEqual(res_int.completed_chunks, 1)
        self.assertLess(len(res_int.audio), len(res_full.audio))
        # Edge taper prevents click spikes at final boundary
        self.assertLess(abs(float(res_int.audio[-1])), 0.05)

        # 3. Pre-cancelled run before synthesis starts
        pre_ev = threading.Event()
        pre_ev.set()
        res_pre = synth.synthesize_stream(text, interrupt_event=pre_ev)
        self.assertTrue(res_pre.interrupted)
        self.assertEqual(res_pre.completed_chunks, 0)
        self.assertEqual(len(res_pre.audio), 0)

        # 4. Subsequent clean synthesis recovers with zero corrupted state
        res_recover = synth.synthesize_stream(text)
        self.assertFalse(res_recover.interrupted)
        self.assertEqual(res_recover.completed_chunks, res_full.total_chunks)
        self.assertEqual(len(res_recover.audio), len(res_full.audio))


@unittest.skipUnless(
    HAS_ORT and os.path.isfile(os.path.join("output", "persian_vits_upgraded.onnx")),
    "Requires onnxruntime + output/persian_vits_upgraded.onnx",
)
class TestRealONNXInterruptAndStress(unittest.TestCase):
    def test_real_onnx_word_stress_and_interrupt(self):
        from models import align_context_embedding

        onnx_path = os.path.join("output", "persian_vits_upgraded.onnx")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        g2p = PersianG2PProsody()
        mana_cfg = os.path.join("pretrained", "mana", "config.json")
        if os.path.isfile(mana_cfg):
            g2p = PersianG2PProsody(piper_config_json_path=mana_cfg)
        else:
            g2p.phoneme_id_map = dict(MANA_ACTIVE_PHONEME_MAP)

        ctx_dim = 512
        for inp in sess.get_inputs():
            if inp.name == "context_emb" and len(inp.shape) == 3 and isinstance(inp.shape[2], int):
                ctx_dim = inp.shape[2]

        def run_onnx_ids(p_ids: List[int], pr_ids: List[int]) -> np.ndarray:
            x = np.expand_dims(np.asarray(p_ids, dtype=np.int64), 0)
            xl = np.array([x.shape[1]], dtype=np.int64)
            pr = np.expand_dims(np.asarray(pr_ids, dtype=np.int64), 0)
            ctx = align_context_embedding(None, x.shape[1], context_dim=ctx_dim)
            ctx_np = ctx.numpy() if hasattr(ctx, "numpy") else np.asarray(ctx, dtype=np.float32)
            scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)
            audio = sess.run(
                None,
                {"input": x, "input_lengths": xl, "prosody_ids": pr, "context_emb": ctx_np, "scales": scales},
            )[0]
            return np.squeeze(audio).astype(np.float32)

        # Word stress variants on real ONNX model
        word_cases = [
            ("hatman_stress1", "h/tm/n", "first_vowel"),
            ("hatman_stress2", "h/tm/n", "second_vowel"),
            ("hatman_none", "h/tm/n", "none"),
            ("hala_stress1", "hala", "first_vowel"),
            ("hala_stress2", "hala", "second_vowel"),
            ("hala_none", "hala", "none"),
        ]
        word_metrics: Dict[str, Any] = {}
        for label, ge2pe_tok, mode in word_cases:
            syms, ids, tags = stress_word_to_streams(g2p, ge2pe_tok, stress_mode=mode)
            wav = run_onnx_ids(ids, tags)
            self.assertGreater(len(wav), 2000)
            self.assertFalse(bool(np.isnan(wav).any() or np.isinf(wav).any()))
            _save_wav_pcm16(os.path.join("output", f"pipeline_stress_{label}.wav"), wav)
            word_metrics[label] = {
                "ipa": "".join(s for s in syms if s not in ("^", "$", "_")),
                **analyze_acoustic_prosody(wav),
            }

        # Real ONNX Interruptible synthesis on multi-clause sentence
        def synth_clause(clause_txt: str) -> np.ndarray:
            _, ids, tags, _, _ = g2p.text_to_aligned_streams_with_context(
                clause_txt, intersperse_blank=True, prefer_homo_ge2pe=False
            )
            return run_onnx_ids(ids, tags)

        ctrl = InterruptibleTTSSynthesizer(synth_clause)
        multi_text = "مدیر جدید در سمت ریاست، حتما با دقت... به سمت اتاق جلسات رفت."
        res_full = ctrl.synthesize_stream(multi_text)
        _save_wav_pcm16(os.path.join("output", "pipeline_interrupt_full.wav"), res_full.audio)

        ev = threading.Event()

        def stop_after_first(idx: int, _tot: int, _txt: str, _chunk: np.ndarray):
            if idx == 0:
                ev.set()

        res_cut = ctrl.synthesize_stream(multi_text, interrupt_event=ev, on_chunk=stop_after_first)
        _save_wav_pcm16(os.path.join("output", "pipeline_interrupt_cut.wav"), res_cut.audio)

        self.assertFalse(res_full.interrupted)
        self.assertTrue(res_cut.interrupted)
        self.assertEqual(res_cut.completed_chunks, 1)
        self.assertLess(len(res_cut.audio), len(res_full.audio))

        report = {
            "onnx_path": onnx_path,
            "context_dim": ctx_dim,
            "word_stress_metrics": word_metrics,
            "interrupt_full": {
                "completed_chunks": res_full.completed_chunks,
                "total_chunks": res_full.total_chunks,
                "interrupted": res_full.interrupted,
                "elapsed_ms": res_full.elapsed_ms,
                **analyze_acoustic_prosody(res_full.audio),
            },
            "interrupt_cut_after_chunk0": {
                "completed_chunks": res_cut.completed_chunks,
                "total_chunks": res_cut.total_chunks,
                "interrupted": res_cut.interrupted,
                "elapsed_ms": res_cut.elapsed_ms,
                **analyze_acoustic_prosody(res_cut.audio),
            },
        }
        with open(os.path.join("output", "pipeline_interrupt_stress_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
