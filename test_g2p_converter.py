"""Unit & Integration Tests for Micro-Level G2P Dataset Converter (Repr. 1 -> Repr. 2)."""

import os
import unittest

from g2p_dataset_converter import (
    DEFAULT_CSV_PATH,
    build_80_20_dataset,
    canonicalize_repr2_for_eval,
    convert_sentence_repr1_to_repr2,
    convert_word_repr1_to_repr2,
)


class TestG2PDatasetConverter(unittest.TestCase):
    def test_16_symbols_initial_at_and_medial_fixes(self):
        self.assertEqual(convert_word_repr1_to_repr2("Catr"), "C/tr")
        self.assertEqual(convert_word_repr1_to_repr2("?omr"), "@omr")
        self.assertEqual(convert_word_repr1_to_repr2("estefAde"), "@estefade")
        self.assertEqual(convert_word_repr1_to_repr2("az"), "@/z")
        self.assertEqual(convert_word_repr1_to_repr2("Sodeand"), "$ode@/nd")
        self.assertEqual(convert_word_repr1_to_repr2("miAyad"), "mi@ay/d")
        self.assertEqual(convert_word_repr1_to_repr2("xeili"), "xeyli")
        self.assertEqual(convert_word_repr1_to_repr2("niAz"), "niyaz")
        self.assertEqual(convert_word_repr1_to_repr2("barrAAq"), "b/rraq")

    def test_ezafe_e1_ye1_enclitic_o_and_trailing_question(self):
        s1, ok1 = convert_sentence_repr1_to_repr2(
            "تیغه اره بسیار تیز است؟", "tiqeye arre besiyAr tiz ast?"
        )
        self.assertTrue(ok1)
        self.assertEqual(s1, "tiqeye1 @/rre besiyar tiz @/st")

        s2, ok2 = convert_sentence_repr1_to_repr2(
            "ابر سیاه و سنگین باران را شروع می‌کند",
            "?abre siyAho sangin bArAn rA Soru? mikonad",
        )
        self.assertTrue(ok2)
        self.assertEqual(s2, "@/bre1 siyah o s/ngin baran ra $oru@ mikon/d")
        self.assertEqual(canonicalize_repr2_for_eval(s1), "tiqeye @/rre besiyar tiz @/st")

    @unittest.skipUnless(os.path.isfile(DEFAULT_CSV_PATH), "Local G2P CSV not present")
    def test_80_20_split_and_audit(self):
        tr, te, sh, summary = build_80_20_dataset(
            DEFAULT_CSV_PATH, train_ratio=0.80, seed=42, audit_out_path="output/g2p_conversion_audit.json"
        )
        self.assertGreater(len(tr), 4000)
        self.assertGreater(len(te), 900)
        self.assertEqual(len(sh), 8)
        tr_sents = {x["input_text"] for x in tr}
        te_sents = {x["input_text"] for x in te}
        self.assertEqual(len(tr_sents & te_sents), 0)
        print("\n[Audit Summary]", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
