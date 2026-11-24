"""
Verification Suite for Persian VITS/Piper Architecture, G2P Stress/Phoneme Alignment,
Zero-Init Guarantees, Full SynthesizerTrn + MultiPeriodDiscriminator Training Step,
and Dynamic-Shape ONNX Export/Runtime Execution.
"""

import os
import tempfile
import unittest

from g2p_persian import PersianG2PProsody
from symbols import PROSODY_TAGS

try:
    import numpy as np
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


MANA_ACTIVE_PHONEME_MAP = {
    "_": 0, "^": 1, "$": 2, " ": 3, "!": 4, ",": 5, "-": 6, ".": 7, ":": 8, ";": 9, "?": 10,
    "a": 14, "b": 15, "d": 17, "e": 18, "f": 19, "h": 20, "i": 21, "j": 22, "k": 23, "l": 24,
    "m": 25, "n": 26, "o": 27, "p": 28, "q": 29, "r": 30, "s": 31, "t": 32, "u": 33, "v": 34,
    "x": 36, "z": 38, "ɑ": 51, "ɡ": 66, "ʃ": 96, "ʒ": 108, "ʔ": 109, "ˈ": 120, "ˌ": 121, "ː": 122, "1": 131,
}


class TestPersianG2PAlignment(unittest.TestCase):
    def setUp(self):
        self.g2p = PersianG2PProsody()
        self.g2p.phoneme_id_map = dict(MANA_ACTIVE_PHONEME_MAP)

    def _word_ipa(self, ge2pe_tok: str, is_q: bool = False) -> str:
        raw = self.g2p._ge2pe_word_to_pairs(ge2pe_tok, is_final_question_word=is_q)
        split = [(sub, tag) for sym, tag in raw for sub in self.g2p._split_ipa_for_piper(sym)]
        for sub, _ in split:
            self.assertIn(sub, self.g2p.phoneme_id_map, f"Unmapped symbol: {sub!r}")
        return "".join(sub for sub, _ in split)

    def test_mana_piper_ipa_and_stress_placement(self):
        self.assertEqual(self._word_ipa("h/tm/n"), "hˈatman")
        self.assertEqual(self._word_ipa("hala"), "hˈɑlɑ")
        self.assertEqual(self._word_ipa("$ir"), "ʃˈir")
        self.assertEqual(self._word_ipa("deqq/t"), "dˈeq1q1at")
        self.assertEqual(self._word_ipa("q/nd"), "q1ˈand")
        self.assertEqual(self._word_ipa("q/za"), "q1azˈɑ")
        self.assertEqual(self._word_ipa("b/rge1"), "bˈarɡe")
        self.assertEqual(self._word_ipa("kerm1"), "kˈerme")
        self.assertEqual(self._word_ipa("k/r/me1"), "karˈame")

    def test_sentence_prosody_tags_and_no_unmapped(self):
        sents = [
            "حتما می‌آیم.",
            "آیا مهر استاد دانشگاه حتما با مهر و محبت بر برگه امتحان زده شد؟",
            "مدیر جدید در سمت ریاست، حتما با دقت به سمت اتاق جلسات رفت.",
        ]
        for text in sents:
            syms, ids, tags, _, _ = self.g2p.text_to_aligned_streams_with_context(
                text, intersperse_blank=True, prefer_homo_ge2pe=False
            )
            self.assertEqual(len(syms), len(ids))
            self.assertEqual(len(ids), len(tags))
            self.assertIn(PROSODY_TAGS["STRESS"], tags)
            self.assertIn("ˈ", syms)
            for sym in syms:
                if sym not in ("^", "$", "_"):
                    self.assertIn(sym, self.g2p.phoneme_id_map, f"Unmapped {sym!r} in {text}")

    def test_normalization_before_g2p(self):
        norm = self.g2p.normalize_persian("امروز 2 کتاب ه.ش دارم؟")
        self.assertIn("دو", norm)
        self.assertIn("هجری شمسی", norm)
        self.assertTrue(norm.endswith("؟"))


@unittest.skipUnless(HAS_TORCH, "PyTorch required for neural architecture tests")
class TestUpgradedPersianVits(unittest.TestCase):
    def test_unit_zero_init_and_flow_invertibility(self):
        from models import (
            DurationPredictorAdapter,
            InvertibleFlowAttentionCoupling,
            ProsodyEnhancedEmbedding,
        )

        torch.manual_seed(42)
        base_emb = nn.Embedding(256, 192)
        up_emb = ProsodyEnhancedEmbedding(base_emb, context_dim=1472)
        x = torch.tensor([[1, 0, 14, 0, 51, 0, 2]], dtype=torch.long)
        p = torch.tensor([[0, 0, 1, 0, 3, 0, 4]], dtype=torch.long)
        ctx = torch.randn(1, 11, 1472)
        with torch.no_grad():
            diff = (up_emb(x, prosody_ids=p, context_emb=ctx) - base_emb(x)).abs().max().item()
        self.assertEqual(diff, 0.0)

        # Verify blank tokens (x == 0) remain unaffected even after non-zero adapter weights
        with torch.no_grad():
            up_emb.prosody_emb.weight.fill_(0.25)
            up_emb.context_proj.weight.fill_(0.05)
            out_after = up_emb(x, prosody_ids=p, context_emb=ctx)
            base_after = base_emb(x)
            blank_mask = x == 0
            blank_diff = (out_after[blank_mask] - base_after[blank_mask]).abs().max().item()
        self.assertEqual(blank_diff, 0.0)

        # Invertible flow attention round-trip check
        flow = InvertibleFlowAttentionCoupling(channels=192, n_heads=4, dropout=0.0).eval()
        with torch.no_grad():
            flow.out_proj.weight.normal_(0.0, 0.1)
            flow.out_proj.bias.normal_(0.0, 0.05)
            z = torch.randn(2, 192, 23)
            z_mask = torch.ones(2, 1, 23)
            z_mask[1, :, 18:] = 0
            z_fwd = flow(z, z_mask, reverse=False)
            z_rec = flow(z_fwd, z_mask, reverse=True)
            inv_err = ((z_rec - z) * z_mask).abs().max().item()
        self.assertLess(inv_err, 1e-5)

        # Duration adapter wrapping Conv1d baseline
        base_dp = nn.Sequential(nn.Conv1d(192, 1, 1))
        dp_adp = DurationPredictorAdapter(base_dp, in_channels=192)
        h = torch.randn(2, 192, 15)
        h_mask = torch.ones(2, 1, 15)
        with torch.no_grad():
            self.assertEqual((dp_adp(h, h_mask) - base_dp(h * h_mask)).abs().max().item(), 0.0)

    def test_real_synthesizer_trn_step0_backward_and_onnx(self):
        from models import (
            UpgradedPersianVits,
            align_context_embedding,
            export_upgraded_vits_onnx,
        )
        from piper_vits import (
            MultiPeriodDiscriminator,
            SynthesizerTrn,
            commons,
            discriminator_loss,
            feature_loss,
            generator_loss,
            kl_loss,
            spectrogram_torch,
        )

        torch.manual_seed(7)
        hop_length = 256
        filter_length = 1024
        seg_frames = 8
        base_g = SynthesizerTrn(
            n_vocab=256,
            spec_channels=filter_length // 2 + 1,
            segment_size=seg_frames,
            inter_channels=192,
            hidden_channels=192,
            filter_channels=384,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.0,
            resblock="2",
            resblock_kernel_sizes=(3, 5),
            resblock_dilation_sizes=((1, 2), (2, 6)),
            upsample_rates=(8, 8, 4),
            upsample_initial_channel=128,
            upsample_kernel_sizes=(16, 16, 8),
            n_speakers=1,
            gin_channels=0,
            use_sdp=True,
        )
        net_d = MultiPeriodDiscriminator(use_spectral_norm=False)

        # Deterministic baseline output before wrapping
        x = torch.tensor([[1, 0, 20, 0, 120, 0, 14, 0, 32, 0, 2, 0]], dtype=torch.long)
        x_len = torch.tensor([x.size(1)], dtype=torch.long)
        base_g.eval()
        with torch.no_grad():
            torch.manual_seed(99)
            base_audio, _, _, _ = base_g.infer(
                x, x_len, noise_scale=0.0, length_scale=1.0, noise_scale_w=0.0
            )

        up_vits = UpgradedPersianVits(base_g, context_dim=1472)
        up_vits.eval()
        p_ids = torch.zeros_like(x)
        ctx = torch.randn(1, 9, 1472)
        with torch.no_grad():
            torch.manual_seed(99)
            up_audio, _, _, _ = up_vits.infer(
                x, x_len, prosody_ids=p_ids, context_emb=ctx,
                noise_scale=0.0, length_scale=1.0, noise_scale_w=0.0
            )
        step0_audio_diff = (up_audio - base_audio).abs().max().item()
        self.assertLess(step0_audio_diff, 1e-6)

        # One real end-to-end VITS + MPD training step
        up_vits.train()
        net_d.train()
        opt_g = torch.optim.AdamW(up_vits.parameters(), lr=2e-4)
        opt_d = torch.optim.AdamW(net_d.parameters(), lr=2e-4)

        wav_len = hop_length * 24
        y_wav = torch.randn(1, wav_len) * 0.1
        spec = spectrogram_torch(y_wav, filter_length, 22050, hop_length, 1024, center=False)
        spec_len = torch.tensor([spec.size(2)], dtype=torch.long)

        y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = up_vits(
            x, x_len, spec, spec_len, prosody_ids=p_ids, context_emb=ctx
        )
        y_real_slice = commons.slice_segments(
            y_wav.unsqueeze(1), ids_slice * hop_length, seg_frames * hop_length
        )

        # D step
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y_real_slice, y_hat.detach())
        loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        opt_d.zero_grad()
        loss_disc.backward()
        opt_d.step()

        # G step
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_real_slice, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        loss_mel = torch.nn.functional.l1_loss(y_hat, y_real_slice)
        loss_g = loss_gen + loss_fm + loss_mel * 15.0 + loss_dur + loss_kl
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        self.assertTrue(torch.isfinite(loss_disc))
        self.assertTrue(torch.isfinite(loss_g))
        self.assertGreater(up_vits.vits.enc_p.emb.prosody_emb.weight.abs().sum().item(), 0.0)
        self.assertGreater(up_vits.vits.dp.out_proj.weight.abs().sum().item(), 0.0)
        self.assertGreater(up_vits.vits.flow.attn_coupling.out_proj.weight.abs().sum().item(), 0.0)

        # Dynamic ONNX export & ORT execution at 2 distinct sequence lengths
        if HAS_ORT:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "test_upgraded.onnx")
                export_upgraded_vits_onnx(up_vits, onnx_path, opset_version=15)
                sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                for dyn_t in (17, 43):
                    dx = np.random.randint(1, 50, size=(1, dyn_t), dtype=np.int64)
                    dx[:, 1::2] = 0
                    dp = np.random.randint(0, 5, size=(1, dyn_t), dtype=np.int64)
                    dctx = align_context_embedding(torch.randn(1, 12, 1472), dyn_t, 1472).numpy()
                    out_wav = sess.run(
                        None,
                        {
                            "input": dx,
                            "input_lengths": np.array([dyn_t], dtype=np.int64),
                            "prosody_ids": dp,
                            "context_emb": dctx,
                            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
                        },
                    )[0]
                    self.assertEqual(out_wav.ndim, 4)
                    self.assertGreater(out_wav.shape[-1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
