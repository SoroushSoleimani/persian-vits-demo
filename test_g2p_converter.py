"""Unit & Integration Tests for Micro-Level G2P Dataset Converter (Repr. 1 -> Repr. 2)."""

import csv
import os
import tempfile
import unittest

from g2p_dataset_converter import (
    DEFAULT_CSV_PATH,
    annotate_sentence_stress_repr2,
    build_80_20_dataset,
    canonicalize_repr2_for_eval,
    convert_sentence_repr1_to_repr2,
    convert_word_repr1_to_repr2,
)
from persian_text_normalizer import normalize_persian_text


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

    def test_persian_text_normalizer_numbers_dates_abbrev_zwnj(self):
        out = normalize_persian_text(
            "د. احمدي در 1403/08/15 با 25% تخفیف می‌رود.",
            use_external=False,
        )
        self.assertIn("دکتر احمدی", out)
        self.assertIn("پانزدهم آبان هزار و چهارصد و سه", out)
        self.assertIn("بیست و پنج درصد", out)
        self.assertIn("می\u200cرود", out)

    def test_stress_homograph_annotation_and_eval(self):
        # ماهی: fish (mah'i) vs monthly (m'ahi)
        s_fish = annotate_sentence_stress_repr2("ماهی قرمز در آب است", "mahi qermez d/r @ab @/st")
        s_monthly = annotate_sentence_stress_repr2("ماهی دو میلیون تومان میگیرد", "mahi do miliyun toman migir/d")
        self.assertIn("mah'i", s_fish)
        self.assertIn("m'ahi", s_monthly)

        # مردی: a man (m'/rdi) vs manliness (m/rd'i)
        s_a_man = annotate_sentence_stress_repr2("مردی از دور آمد", "m/rdi @/z dur @am/d")
        s_manliness = annotate_sentence_stress_repr2("از مردی و غیرت او گفتند", "@/z m/rdi o qeyr/te1 @u goft/nd")
        self.assertIn("m'/rdi", s_a_man)
        self.assertIn("m/rd'i", s_manliness)

        # ولی: conjunction (v'/li) vs guardian (v/l'i)
        s_but = annotate_sentence_stress_repr2("ولی او نیامد", "v/li @u n/yam/d")
        s_guardian = annotate_sentence_stress_repr2("ولی فقیه", "v/li1 f/qih")
        self.assertIn("v'/li", s_but)
        self.assertIn("v/l'i1", s_guardian)

        # canonicalize_repr2_for_eval: keep_stress=False strips ', keep_stress=True preserves '
        self.assertEqual(canonicalize_repr2_for_eval("m'ahi", keep_stress=False), "mahi")
        self.assertEqual(canonicalize_repr2_for_eval("m'ahi", keep_stress=True), "m'ahi")

    def test_word_lemma_split_disjoint(self):
        rows = [
            ("کرم مرد بزرگ", "karame marde bozorg", "کرم", "karam"),
            ("کرم کوچک سبز", "kerme kucak sabz", "کرم", "kerm"),
            ("مهر استاد دانشگاه", "mohre ostad danesgah", "مهر", "mohr"),
            ("مهر و محبت", "mehr o mohabbat", "مهر", "mehr"),
            ("سمت راست خیابان", "samte rast xiyaban", "سمت", "samt"),
            ("سمت مدیریت", "semate modiriyat", "سمت", "semat"),
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", encoding="utf-8", newline="", delete=False) as tf:
            writer = csv.writer(tf)
            writer.writerow(["grapheme", "phoneme", "homograph_grapheme", "homograph_phoneme"])
            writer.writerows(rows)
            tmp_path = tf.name
        try:
            tr, te, _, summary = build_80_20_dataset(tmp_path, train_ratio=0.67, seed=42, split_mode="word_lemma")
            self.assertEqual(summary["split_mode"], "word_lemma")
            tr_hw = {x["homograph_grapheme"] for x in tr}
            te_hw = {x["homograph_grapheme"] for x in te}
            self.assertGreater(len(tr_hw), 0)
            self.assertGreater(len(te_hw), 0)
            self.assertEqual(len(tr_hw & te_hw), 0)
        finally:
            os.remove(tmp_path)

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
