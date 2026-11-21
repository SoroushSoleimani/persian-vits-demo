"""Comprehensive End-to-End Test Suite for Persian TTS Pipeline (VITS + Piper Base).

Covers every block of the architecture:
  - Block 1: Text Normalization (`persian_normalizer`)
  - Block D: Homograph Dataset (`homograph_dataset_pipeline`, Repr.1->Repr.2, Word/Lemma split, Extra Real TTS set)
  - Block 2-4: Persian G2P + Homograph Fix + Phoneme IDs + Prosody Tags (`g2p_persian`)
  - Block 5-8 & T1-T3: VITS Prior/TextEncoder, DurationPredictor+MAS, Flow, HiFi-GAN Decoder,
    PosteriorEncoder, MultiPeriod+MultiScale Discriminator, Checkpoint Init & ONNX Export (`models`)
  - Block 9: Export + Inference (`tts_pipeline_module`: chunking, `scales`, silence trim, interrupt)
"""

import json
import os
import threading
import unittest
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from g2p_persian import PersianG2PProsody
from homograph_dataset_pipeline import (
    DEFAULT_CSV_PATH,
    EXTRA_REAL_TTS_TEST_SET,
    convert_dataset_phoneme_to_repr2,
    load_homograph_dataset_by_word_lemma,
    repr1_word_to_repr2,
)
from persian_normalizer import normalize_persian_text
from symbols import BOS, EOS, PAD, PROSODY_TAGS
from test_pipeline import MANA_ACTIVE_PHONEME_MAP
from tts_pipeline_module import (
    TARGET_SR,
    PersianTTSPipeline,
    chunk_text_for_tts,
    compute_audio_metrics,
    trim_and_normalize_audio,
)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def _save_wav(path: str, wav: np.ndarray, sr: int = TARGET_SR) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class TestBlock1TextNormalization(unittest.TestCase):
    def test_numbers_dates_abbreviations_and_zwnj(self):
        raw = "تاريخ جلسه ۱۴۰۳/۰۸/۲۵ با د. رضايي در مسافت 5 km و ۲۵٪ پیشرفت می‌رود."
        norm = normalize_persian_text(raw)
        self.assertIn("بیست و پنجم آبان هزار و چهارصد و سه", norm)
        self.assertIn("دکتر رضایی", norm)
        self.assertIn("پنج کیلومتر", norm)
        self.assertIn("بیست و پنج درصد", norm)
        self.assertIn("می\u200cرود", norm)  # ZWNJ preserved
        self.assertNotIn("ي", norm)

    def test_decimals_and_zero(self):
        self.assertEqual(normalize_persian_text("سرعت ۳.۵ و عدد ۰"), "سرعت سه ممیز پنج و عدد صفر")


class TestBlockDDatasetAndRepr2(unittest.TestCase):
    def test_repr1_to_repr2_mapping(self):
        self.assertEqual(repr1_word_to_repr2("Catr"), "C/tr")
        self.assertEqual(repr1_word_to_repr2("?omr"), "@omr")
        self.assertEqual(repr1_word_to_repr2("Sahr"), "$/hr")
        self.assertEqual(repr1_word_to_repr2("ZAle"), ";ale")
        self.assertEqual(repr1_word_to_repr2("mAh"), "mah")
        self.assertEqual(repr1_word_to_repr2("dard"), "d/rd")
        self.assertEqual(
            convert_dataset_phoneme_to_repr2("تیغه اره تیز است؟", "tiqeye arre tiz ?ast?"),
            "tiqeye /rre tiz @/st",
        )

    @unittest.skipUnless(os.path.isfile(DEFAULT_CSV_PATH), "Local homograph CSV not present")
    def test_word_lemma_split_and_extra_tts_set(self):
        tr, te, extra = load_homograph_dataset_by_word_lemma(DEFAULT_CSV_PATH, 300, 80, seed=42)
        self.assertGreater(len(tr), 100)
        self.assertGreater(len(te), 20)
        tr_sents = {x["grapheme"] for x in tr}
        te_sents = {x["grapheme"] for x in te}
        self.assertEqual(len(tr_sents & te_sents), 0)
        self.assertEqual(len(extra), 6)


class TestBlock234G2PAndProsody(unittest.TestCase):
    def setUp(self):
        self.g2p = PersianG2PProsody()
        self.g2p.phoneme_id_map = dict(MANA_ACTIVE_PHONEME_MAP)

    def test_uppercase_c_and_tashdid_and_prosody_streams(self):
        pairs = self.g2p._ge2pe_word_to_pairs("C/tr")
        self.assertEqual("".join(s for s, _ in pairs), "tʃˈatr")

        syms, ids, tags, _, _ = self.g2p.text_to_aligned_streams_with_context(
            "آیا معلم مدرسه با دقت خواند؟", intersperse_blank=True, prefer_homo_ge2pe=False
        )
        self.assertEqual(syms[0], PAD)
        self.assertEqual(syms[1], BOS)
        self.assertEqual(syms[-2], EOS)
        self.assertEqual(syms[-1], PAD)
        self.assertEqual(len(syms), len(ids))
        self.assertEqual(len(ids), len(tags))
        self.assertIn(PROSODY_TAGS["TASHDID"], tags)
        self.assertIn(PROSODY_TAGS["STRESS"], tags)
        self.assertIn(PROSODY_TAGS["EZAFE"], tags)
        self.assertIn(PROSODY_TAGS["QUESTION"], tags)


@unittest.skipUnless(HAS_TORCH, "PyTorch required for VITS & Discriminator tests")
class TestBlock5678T1T2T3VITSArchitecture(unittest.TestCase):
    def test_full_vits_blocks_and_discriminators(self):
        from models import UpgradedPersianVits, attach_upgrades_to_piper_vits
        from piper_vits import MultiPeriodDiscriminator, SynthesizerTrn

        base_vits = SynthesizerTrn(
            n_vocab=64,
            spec_channels=513,
            segment_size=32,
            inter_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            n_speakers=0,
            gin_channels=0,
            use_sdp=True,
        )
        mpd = MultiPeriodDiscriminator()  # DiscriminatorS (scale) + DiscriminatorP (periods)
        self.assertEqual(len(mpd.discriminators), 6)  # 1 Multi-Scale + 5 Multi-Period

        up = UpgradedPersianVits(attach_upgrades_to_piper_vits(base_vits, context_dim=512))
        up.train()
        x = torch.randint(1, 40, (1, 16), dtype=torch.long)
        xl = torch.tensor([16], dtype=torch.long)
        spec = torch.randn(1, 513, 24)
        sl = torch.tensor([24], dtype=torch.long)
        pr = torch.randint(0, 5, (1, 16), dtype=torch.long)
        ctx = torch.randn(1, 16, 512)

        y_hat, l_dur, attn, ids_sl, x_m, z_m, (z, z_p, m_p, logs_p, m_q, logs_q) = up(
            x, xl, spec, sl, prosody_ids=pr, context_emb=ctx
        )
        self.assertEqual(y_hat.shape[0], 1)
        self.assertGreater(y_hat.shape[-1], 0)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(y_hat, y_hat.detach())
        self.assertEqual(len(y_d_rs), 6)
        loss = y_hat.abs().mean() + l_dur.mean() + sum(g.abs().mean() for g in y_d_gs)
        loss.backward()


@unittest.skipUnless(
    HAS_ORT and os.path.isfile(os.path.join("output", "persian_vits_upgraded.onnx")),
    "Requires onnxruntime + output/persian_vits_upgraded.onnx",
)
class TestBlock9InferenceAndRealTTS(unittest.TestCase):
    def test_end_to_end_real_tts_scales_and_interrupt(self):
        from models import align_context_embedding

        onnx_path = os.path.join("output", "persian_vits_upgraded.onnx")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        mana_cfg = os.path.join("pretrained", "mana", "config.json")
        g2p = PersianG2PProsody(piper_config_json_path=mana_cfg) if os.path.isfile(mana_cfg) else PersianG2PProsody()
        if not os.path.isfile(mana_cfg):
            g2p.phoneme_id_map = dict(MANA_ACTIVE_PHONEME_MAP)

        ge2pe_dir = os.path.join("pretrained", "homo-ge2pe")
        prefer_ge2pe = False
        if os.path.isdir(ge2pe_dir):
            prefer_ge2pe = g2p.attach_homo_ge2pe(ge2pe_dir, gpu=False)

        ctx_dim = 512
        for inp in sess.get_inputs():
            if inp.name == "context_emb" and len(inp.shape) == 3 and isinstance(inp.shape[2], int):
                ctx_dim = inp.shape[2]

        def synth_ids(
            p_ids: List[int], pr_ids: List[int], ctx_emb: Optional[Any], scales: Tuple[float, float, float]
        ) -> np.ndarray:
            if hasattr(ort, "set_seed"):
                ort.set_seed(42)
            x = np.expand_dims(np.asarray(p_ids, dtype=np.int64), 0)
            xl = np.array([x.shape[1]], dtype=np.int64)
            pr = np.expand_dims(np.asarray(pr_ids, dtype=np.int64), 0)
            ctx = align_context_embedding(ctx_emb, x.shape[1], context_dim=ctx_dim)
            ctx_np = ctx.numpy() if hasattr(ctx, "numpy") else np.asarray(ctx, dtype=np.float32)
            sc = np.asarray(scales, dtype=np.float32)
            audio = sess.run(
                None,
                {"input": x, "input_lengths": xl, "prosody_ids": pr, "context_emb": ctx_np, "scales": sc},
            )[0]
            return np.squeeze(audio).astype(np.float32)

        pipe = PersianTTSPipeline(g2p=g2p, synth_ids_fn=synth_ids, prefer_homo_ge2pe=prefer_ge2pe)

        # 1. Synthesize all 6 Extra Real TTS sentences (with dates, abbreviations, numbers, homographs, tashdid)
        tts_results: List[Dict[str, Any]] = []
        for item in EXTRA_REAL_TTS_TEST_SET:
            res = pipe.synthesize(item["text"], scales=(0.5, 1.0, 0.35))
            self.assertFalse(res.has_nan_or_inf)
            metrics = compute_audio_metrics(res.audio)
            self.assertGreater(metrics["rms_db"], -25.0)
            self.assertGreater(metrics["mean_f0_hz"], 130.0)
            self.assertLess(metrics["silence_ratio"], 0.38)
            p_syms, _, _, _, raw_g = g2p.text_to_aligned_streams_with_context(
                res.normalized_full_text, intersperse_blank=False, prefer_homo_ge2pe=prefer_ge2pe
            )
            wav_path = os.path.join("output", f"full_pipe_{item['id']}.wav")
            _save_wav(wav_path, res.audio)
            tts_results.append({
                "id": item["id"],
                "raw_text": item["text"],
                "normalized_text": res.normalized_full_text,
                "raw_ge2pe": raw_g,
                "ipa_preview": "".join(s for s in p_syms if s not in ("^", "$", "_")),
                "focus": item["focus"],
                "wav_path": wav_path,
                "elapsed_ms": res.elapsed_ms,
                **metrics,
            })

        # 2. Verify `scales` length_scale control (fast < normal < slow)
        scale_metrics: Dict[str, Any] = {}
        scale_sample = "معلم مدرسه بیش از ۱۲۰ صفحه کتاب را می‌خواند."
        for lbl, sc in (("fast", (0.5, 0.85, 0.35)), ("normal", (0.5, 1.0, 0.35)), ("slow", (0.5, 1.25, 0.35))):
            r = pipe.synthesize(scale_sample, scales=sc)
            m = compute_audio_metrics(r.audio)
            _save_wav(os.path.join("output", f"full_pipe_scale_{lbl}.wav"), r.audio)
            scale_metrics[lbl] = {"scales": list(sc), **m}
        self.assertLess(scale_metrics["fast"]["duration_sec"], scale_metrics["slow"]["duration_sec"])

        # 3. Verify chunking + mid-utterance interrupt
        multi = "مدیر جدید در سمت ریاست، گزارش ۲۵٪ بودجه را بررسی کرد... سپس در تاریخ ۱۴۰۳/۰۸/۲۵ نتیجه را اعلام کرد."
        r_full = pipe.synthesize(multi)
        _save_wav(os.path.join("output", "full_pipe_interrupt_full.wav"), r_full.audio)
        ev = threading.Event()
        ev.set()
        r_pre = pipe.synthesize(multi, interrupt_event=ev)
        self.assertTrue(r_pre.interrupted)
        self.assertEqual(r_pre.completed_chunks, 0)

        report = {
            "onnx_path": onnx_path,
            "context_dim": ctx_dim,
            "homo_ge2pe_attached": prefer_ge2pe,
            "extra_real_tts_results": tts_results,
            "scale_sweep": scale_metrics,
            "multi_chunk_full": {
                "chunks": r_full.total_chunks,
                "normalized": r_full.normalized_full_text,
                **compute_audio_metrics(r_full.audio),
            },
        }
        with open(os.path.join("output", "full_pipeline_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
