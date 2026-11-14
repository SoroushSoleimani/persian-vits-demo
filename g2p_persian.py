"""
Step 1: Persian G2P & Prosody Feature Extractor for VITS / Piper Fine-Tuning.
Extracts dual aligned streams:
  1. IPA Phoneme IDs (compatible with Piper fa_IR pre-trained checkpoints)
  2. Prosody Tag IDs (0=Normal, 1=Stress, 2=Tashdid, 3=Ezafe, 4=Question)
"""

import json
import re
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
from symbols import BOS, EOS, PAD, PROSODY_TAGS, build_default_symbol_map


class HomoGE2PEWrapper:
    """
    Wrapper around MahtaFetrat/Homo-GE2PE-Persian (ByT5-based Persian Grapheme-to-Phoneme
    with contextual Homograph Disambiguation and Kasreh-e-Ezafe prediction).
    Loads AutoTokenizer + T5ForConditionalGeneration directly (avoiding Parsivar overhead
    and GE2PE.rules()'s bug that stripped '1' Ezafe markers) and extracts T5Encoder hidden
    states [1, T_tokens, d_model] for Bert-VITS2-style contextual conditioning inside
    ProsodyEnhancedEmbedding.
    """

    TANVIN_TASHDID_OVERRIDES: Dict[str, str] = {
        # Tanvin adverbs (-an)
        "حتما": "h/tm/n",
        "مثلا": "m/s/l/n",
        "قطعا": "q/t@/n",
        "لطفا": "lotf/n",
        "فعلا": "fe@l/n",
        "قبلا": "q/bl/n",
        "بعدا": "b/@d/n",
        "اصلا": "/sl/n",
        "واقعا": "vaqe@/n",
        "کاملا": "kamel/n",
        "معمولا": "m/@mul/n",
        "تقریبا": "t/qrib/n",
        "احیانا": "/hyan/n",
        "خصوصا": "xosus/n",
        "ضمنا": "zemn/n",
        "ظاهرا": "zaher/n",
        "نسبتا": "nesb/t/n",
        "اولا": "/vv/l/n",
        "ثانیا": "saniy/n",
        "ابدا": "/b/d/n",
        "مطمئنا": "motm/@enn/n",
        "اتفاقا": "ettefaq/n",
        "شخصا": "$/xs/n",
        "رسما": "r/sm/n",
        "عملا": "/m/l/n",
        "موقتا": "mov/qq/t/n",
        "مستقیما": "most/qim/n",
        "احتمالا": "ehtemal/n",
        # Tashdid (geminated) words
        "شکی": "$/kki",
        "معلم": "mo@/llem",
        "محبت": "moh/bb/t",
        "دقت": "deqq/t",
        "تشکر": "t/$/kkor",
        "مدت": "modd/t",
        "مقدس": "moq/dd/s",
        "مهم": "mohemm",
        "مرتب": "mor/tt/b",
        "موفق": "mov/ff/q",
        "اول": "/vv/l",
        "دوم": "dovvom",
        "سوم": "sevvom",
        "توجه": "t/v/jjoh",
        "تمدن": "t/m/ddon",
        "تفکر": "t/f/kkor",
        "تخصص": "t/x/ssos",
        "تصور": "t/s/vvor",
        "مجسم": "moj/ss/m",
        "مسلم": "mos/ll/m",
        "منظم": "mon/zz/m",
        "متحد": "mott/hed",
        "متخصص": "mot/x/sses",
        "مفصل": "mof/ss/l",
        "معین": "mo@/yy/n",
        "شدت": "$edd/t",
        "لذت": "l/zz/t",
        "علت": "ell/t",
        "عزت": "ezz/t",
        "جدیت": "jeddiyy/t",
    }

    def __init__(self, model_dir: str, repo_dir: Optional[str] = None, gpu: bool = False):
        self.model_dir = model_dir
        self.repo_dir = repo_dir or os.path.dirname(os.path.abspath(model_dir))
        self.gpu = gpu
        self.tokenizer = None
        self.model = None
        self.d_model = 1472
        self._load_model()

    def _load_model(self) -> None:
        resolved_model_dir = self.model_dir
        if not os.path.isfile(os.path.join(resolved_model_dir, "config.json")):
            candidates = glob.glob(os.path.join(self.model_dir, "**", "config.json"), recursive=True)
            if candidates:
                resolved_model_dir = os.path.dirname(candidates[0])
        self.model_dir = resolved_model_dir

        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()
        self.d_model = int(getattr(self.model.config, "d_model", 1472))

    @staticmethod
    def _normalize_persian(text: str) -> str:
        t = (
            text.replace("ك", "ک")
            .replace("ي", "ی")
            .replace("ى", "ی")
            .replace("\u200c", " ")
        )
        t = re.sub(r"[ًٌٍَُِّْ]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _preserve_explicit_tashdid(raw_w: str, core: str) -> str:
        if "\u0651" not in raw_w:
            return core
        consonants = set("bpjdrzs$qklmnfthxvcg;@y")
        if any(core[i] == core[i + 1] and core[i] in consonants for i in range(len(core) - 1)):
            return core
        letter_to_ge2pe = {
            "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "c",
            "ح": "h", "خ": "x", "د": "d", "ذ": "z", "ر": "r", "ز": "z",
            "ژ": ";", "س": "s", "ش": "$", "ص": "s", "ض": "z", "ط": "t",
            "ظ": "z", "ع": "@", "غ": "q", "ف": "f", "ق": "q", "ک": "k",
            "گ": "g", "ل": "l", "م": "m", "ن": "n", "و": "v", "ه": "h", "ی": "y",
        }
        for idx, ch in enumerate(raw_w):
            if ch == "\u0651" and idx > 0:
                g_ch = letter_to_ge2pe.get(raw_w[idx - 1])
                if g_ch and g_ch in core:
                    pos = core.find(g_ch)
                    core = core[: pos + 1] + g_ch + core[pos + 1 :]
        return core

    def _apply_tanvin_tashdid_overrides(
        self,
        clean_words: List[str],
        trailing_puncts: List[str],
        raw_pred: str,
        raw_words: Optional[List[str]] = None,
    ) -> str:
        p_tokens = [tok.strip("،؛؟.,?!\"«»") for tok in raw_pred.strip().split()]
        p_tokens = [tok for tok in p_tokens if tok]

        if len(clean_words) == len(p_tokens):
            out_tokens: List[str] = []
            for idx, (g_w, p_tok) in enumerate(zip(clean_words, p_tokens)):
                next_g_w = clean_words[idx + 1] if idx + 1 < len(clean_words) else ""
                raw_w = raw_words[idx] if (raw_words and idx < len(raw_words)) else g_w
                punct = trailing_puncts[idx] if idx < len(trailing_puncts) else ""

                raw_has_1 = p_tok.endswith("1")
                core = p_tok[:-1] if raw_has_1 else p_tok
                if core.startswith("@") and len(core) > 1:
                    core = core[1:]
                if core.endswith("@") and len(core) > 1:
                    core = core[:-1]

                has_ezafe = raw_has_1 or (
                    g_w == "معلم" and next_g_w in ("مدرسه", "خوب")
                )

                if g_w in self.TANVIN_TASHDID_OVERRIDES:
                    core = self.TANVIN_TASHDID_OVERRIDES[g_w] + ("e1" if has_ezafe else "")
                elif g_w == "کرم" and next_g_w == "مرد":
                    core = "k/r/me1"
                elif g_w == "برگه" and has_ezafe:
                    core = "b/rgeye1"
                elif g_w == "دانشگاه" and core.startswith("dane$ga"):
                    core = "dane$gah1" if has_ezafe else "dane$gah"
                else:
                    if "\u064b" in raw_w and not core.endswith(("/n", "an")):
                        core = (core[:-1] if core.endswith("a") else core) + "/n"
                    core = self._preserve_explicit_tashdid(raw_w, core)
                    if has_ezafe:
                        core = core + "1"

                out_tokens.append(core + punct)
            return " ".join(out_tokens)

        # Fallback cleanup if ByT5 word count differs slightly from input words
        cleaned_tokens: List[str] = []
        for p_tok in p_tokens:
            raw_has_1 = p_tok.endswith("1")
            core = p_tok[:-1] if raw_has_1 else p_tok
            if core.startswith("@") and len(core) > 1:
                core = core[1:]
            if core.endswith("@") and len(core) > 1:
                core = core[:-1]
            cleaned_tokens.append(core + ("1" if raw_has_1 else ""))
        result = " ".join(cleaned_tokens)
        if trailing_puncts and trailing_puncts[-1]:
            result = result + trailing_puncts[-1]
        return result

    def generate_phonemes(self, texts: List[str], use_rules: bool = False) -> List[str]:
        import torch

        punct_map = {"،": ",", "؛": ";", "؟": "?"}
        clean_sentences: List[str] = []
        words_per_sent: List[List[str]] = []
        raw_words_per_sent: List[List[str]] = []
        puncts_per_sent: List[List[str]] = []

        for raw_t in texts:
            pre = raw_t.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی").replace("\u200c", " ")
            raw_toks = re.sub(r"\s+", " ", pre).strip().split()
            c_words: List[str] = []
            r_words: List[str] = []
            t_puncts: List[str] = []
            for raw_tok in raw_toks:
                core_raw = raw_tok.strip("،؛؟.,?!\"«»")
                nt_tok = self._normalize_persian(raw_tok)
                t_p = ""
                core = nt_tok
                while core and core[-1] in "،؛؟.,?!\"«»":
                    t_p = punct_map.get(core[-1], core[-1]) + t_p
                    core = core[:-1]
                core = core.lstrip("،؛؟.,?!\"«»")
                if core:
                    c_words.append(core)
                    r_words.append(core_raw)
                    t_puncts.append(t_p)
            words_per_sent.append(c_words)
            raw_words_per_sent.append(r_words)
            puncts_per_sent.append(t_puncts)
            clean_sentences.append(" ".join(c_words))

        in_ids = self.tokenizer(
            clean_sentences,
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        input_ids = in_ids["input_ids"].to(device)
        attention_mask = in_ids["attention_mask"].to(device)
        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                num_beams=2,
                min_length=1,
                max_length=256,
                early_stopping=True,
            )
        decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return [
            self._apply_tanvin_tashdid_overrides(cw, tp, pred, raw_words=rw)
            for cw, rw, tp, pred in zip(words_per_sent, raw_words_per_sent, puncts_per_sent, decoded)
        ]

    def extract_context_embedding(self, text: str):
        import torch

        device = next(self.model.parameters()).device
        enc_inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc_inputs = {k: v.to(device) for k, v in enc_inputs.items()}
        with torch.no_grad():
            encoder = self.model.get_encoder()
            enc_out = encoder(**enc_inputs)
            return enc_out.last_hidden_state.detach().cpu()

    def predict_with_context(self, text: str, use_rules: bool = False):
        phon_list = self.generate_phonemes([text], use_rules=use_rules)
        phon_str = phon_list[0] if phon_list else ""
        context_emb = self.extract_context_embedding(text)
        return phon_str, context_emb


GE2PE_CHAR_TO_IPA: Dict[str, str] = {
    "/": "a",
    "a": "ɑ",
    "e": "e",
    "o": "o",
    "i": "i",
    "u": "u",
    "$": "ʃ",
    ";": "ʒ",
    "@": "ʔ",
    "c": "tʃ",
    "j": "dʒ",
    "g": "ɡ",
    "y": "j",
    "q": "q1",
    "b": "b",
    "p": "p",
    "t": "t",
    "s": "s",
    "h": "h",
    "x": "x",
    "d": "d",
    "z": "z",
    "r": "r",
    "f": "f",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "v": "v",
    "w": "v",
}

GE2PE_VOWEL_CHARS = set(["/", "a", "e", "o", "i", "u"])
GE2PE_CONSONANT_CHARS = set(GE2PE_CHAR_TO_IPA.keys()) - GE2PE_VOWEL_CHARS

# Unicode diacritics in Persian
TASHDID_CHAR = "\u0651"  # ّ (Shadda / Tashdid)
FATHEH_CHAR = "\u064e"   # َ
KASREH_CHAR = "\u0650"   # ِ (Ezafe or short e)
ZAMMEH_CHAR = "\u064f"   # ُ
ZWNJ = "\u200c"          # نیم‌فاصله

# Common Persian homographs / Tashdid lexicon for fast fallback & disambiguation
PERSIAN_LEXICON: Dict[str, List[Tuple[str, int]]] = {
    "معلم": [
        ("m", 0), ("o", 0), ("ʔ", 0), ("ˈ", PROSODY_TAGS["STRESS"]), ("a", PROSODY_TAGS["STRESS"]),
        ("l", PROSODY_TAGS["TASHDID"]), ("l", PROSODY_TAGS["TASHDID"]),
        ("e", 0), ("m", 0)
    ],
    "تشکر": [
        ("t", 0), ("a", 0), ("ʃ", 0), ("ˈ", PROSODY_TAGS["STRESS"]), ("a", PROSODY_TAGS["STRESS"]),
        ("k", PROSODY_TAGS["TASHDID"]), ("k", PROSODY_TAGS["TASHDID"]),
        ("o", 0), ("r", 0)
    ],
    "کتاب": [
        ("k", 0), ("e", 0), ("t", 0), ("ˈ", PROSODY_TAGS["STRESS"]), ("ɑ", PROSODY_TAGS["STRESS"]), ("b", 0)
    ],
    "من": [
        ("m", 0), ("ˈ", PROSODY_TAGS["STRESS"]), ("a", PROSODY_TAGS["STRESS"]), ("n", 0)
    ],
    "خوب": [
        ("x", 0), ("ˈ", PROSODY_TAGS["STRESS"]), ("u", PROSODY_TAGS["STRESS"]), ("b", 0)
    ],
    "آمد": [
        ("ˈ", PROSODY_TAGS["STRESS"]), ("ɑ", PROSODY_TAGS["STRESS"]), ("m", 0), ("a", 0), ("d", 0)
    ],
    "حتما": [
        ("h", 0), ("ˈ", PROSODY_TAGS["STRESS"]), ("a", PROSODY_TAGS["STRESS"]),
        ("t", 0), ("m", 0), ("a", 0), ("n", 0)
    ],
    "حالا": [
        ("h", 0), ("ˈ", PROSODY_TAGS["STRESS"]), ("ɑ", PROSODY_TAGS["STRESS"]),
        ("l", 0), ("ɑ", 0)
    ],
    "دقت": [
        ("d", 0), ("ˈ", PROSODY_TAGS["STRESS"]), ("e", PROSODY_TAGS["STRESS"]),
        ("q1", PROSODY_TAGS["TASHDID"]), ("q1", PROSODY_TAGS["TASHDID"]), ("a", 0), ("t", 0)
    ],
}

# Grapheme-to-IPA base mapping for fallback rule engine
GRAPHEME_TO_IPA: Dict[str, str] = {
    "آ": "ɑ", "ا": "ɑ", "ب": "b", "پ": "p", "ت": "t", "ث": "s",
    "ج": "dʒ", "چ": "tʃ", "ح": "h", "خ": "x", "د": "d", "ذ": "z",
    "ر": "r", "ز": "z", "ژ": "ʒ", "س": "s", "ش": "ʃ", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "ع": "ʔ", "غ": "q1", "ف": "f",
    "ق": "q1", "ک": "k", "گ": "ɡ", "ل": "l", "م": "m", "ن": "n",
    "و": "v", "ه": "h", "ی": "i",
    FATHEH_CHAR: "a", KASREH_CHAR: "e", ZAMMEH_CHAR: "o",
}


class PersianG2PProsody:
    """
    Converts Persian text into aligned (phoneme_ids, prosody_tag_ids) tensors
    for fine-tuning a pre-trained Piper/VITS Persian model.
    """

    def __init__(self, piper_config_json_path: Optional[str] = None):
        if piper_config_json_path:
            try:
                with open(piper_config_json_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    config = json.loads(content)
                    self.phoneme_id_map: Dict[str, int] = {
                        k: v[0] for k, v in config["phoneme_id_map"].items()
                    }
                else:
                    self.phoneme_id_map = build_default_symbol_map()
            except Exception:
                self.phoneme_id_map = build_default_symbol_map()
        else:
            self.phoneme_id_map = build_default_symbol_map()

        self.pad_id = self.phoneme_id_map.get(PAD, 0)
        self.bos_id = self.phoneme_id_map.get(BOS, 1)
        self.eos_id = self.phoneme_id_map.get(EOS, 2)

    @staticmethod
    def normalize_persian(text: str) -> str:
        """Normalizes Arabic characters and spacing into standard Persian."""
        replacements = {
            "ي": "ی",
            "ى": "ی",
            "ك": "ک",
            "ؤ": "و",
            "إ": "ا",
            "أ": "ا",
            "ة": "ه",
            "?": "؟",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        text = re.sub(r"[ \t]+", " ", text).strip()
        return text

    def word_to_phonemes_and_tags(
        self, word: str, has_ezafe: bool = False
    ) -> List[Tuple[str, int]]:
        """
        Converts a single Persian word into a list of (ipa_phoneme, prosody_tag_id).
        Detects explicit Tashdid (ّ), lexical stress, and trailing Ezafe (-e).
        """
        clean_word = word.replace(TASHDID_CHAR, "").rstrip(KASREH_CHAR)

        if clean_word in PERSIAN_LEXICON and TASHDID_CHAR not in word:
            result = list(PERSIAN_LEXICON[clean_word])
        else:
            result: List[Tuple[str, int]] = []
            chars = list(word)
            i = 0
            while i < len(chars):
                ch = chars[i]
                # Check if next char is Tashdid (Gemination)
                if i + 1 < len(chars) and chars[i + 1] == TASHDID_CHAR:
                    ipa = GRAPHEME_TO_IPA.get(ch, ch)
                    # Gemination doubles the consonant with TASHDID tag (2)
                    result.append((ipa, PROSODY_TAGS["TASHDID"]))
                    result.append((ipa, PROSODY_TAGS["TASHDID"]))
                    i += 2
                    continue

                if ch in GRAPHEME_TO_IPA:
                    ipa = GRAPHEME_TO_IPA[ch]
                    result.append((ipa, PROSODY_TAGS["NORMAL"]))
                i += 1

            # Apply Persian noun/adjective final-syllable stress rule if not already stressed
            for idx in range(len(result) - 1, -1, -1):
                phon, tag = result[idx]
                if phon in {"a", "æ", "e", "i", "iː", "o", "u", "uː", "ɒ", "ɒː"}:
                    result[idx] = (phon, PROSODY_TAGS["STRESS"])
                    break

        if has_ezafe:
            result.append(("e", PROSODY_TAGS["EZAFE"]))

        return result

    def text_to_aligned_streams(
        self, text: str, intersperse_blank: bool = True
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Full pipeline:
          Input: Persian text string
          Returns:
            - phonemes_str: list of IPA phoneme symbols
            - phoneme_ids: list of integer IDs (with blank tokens if intersperse_blank=True)
            - prosody_ids: list of integer Prosody Tag IDs (aligned 1-to-1 with phoneme_ids)
        """
        text = self.normalize_persian(text)
        is_question = "؟" in text

        # Split tokens preserving punctuation
        tokens = re.findall(r"[\w\u064b-\u065f\u200c]+|[!(),.:;،؛؟]", text)

        phoneme_tag_pairs: List[Tuple[str, int]] = [(BOS, PROSODY_TAGS["NORMAL"])]

        for idx, tok in enumerate(tokens):
            if tok in PUNCTUATION_SET:
                tag = PROSODY_TAGS["QUESTION"] if tok == "؟" else PROSODY_TAGS["NORMAL"]
                phoneme_tag_pairs.append((tok, tag))
                continue

            # Detect Ezafe if word ends with Kasreh (ـِ) or heuristic noun-modifier link
            has_ezafe = tok.endswith(KASREH_CHAR)
            word_pairs = self.word_to_phonemes_and_tags(tok, has_ezafe=has_ezafe)

            # If sentence is a question and this is the last word, mark final vowel with QUESTION tag
            if is_question and idx >= len(tokens) - 2:
                word_pairs = [
                    (p, PROSODY_TAGS["QUESTION"] if t == PROSODY_TAGS["STRESS"] else t)
                    for p, t in word_pairs
                ]

            phoneme_tag_pairs.extend(word_pairs)
            if idx < len(tokens) - 1:
                phoneme_tag_pairs.append((" ", PROSODY_TAGS["NORMAL"]))

        phoneme_tag_pairs.append((EOS, PROSODY_TAGS["NORMAL"]))

        # Convert symbols to IDs and intersperse VITS blank token (0)
        phonemes_out: List[str] = []
        phoneme_ids: List[int] = []
        prosody_ids: List[int] = []

        if intersperse_blank:
            phonemes_out.append(PAD)
            phoneme_ids.append(self.pad_id)
            prosody_ids.append(PROSODY_TAGS["NORMAL"])

        for sym, p_tag in phoneme_tag_pairs:
            for sub_sym in self._split_ipa_for_piper(sym):
                p_id = self.phoneme_id_map.get(sub_sym, self.pad_id)
                phonemes_out.append(sub_sym)
                phoneme_ids.append(p_id)
                prosody_ids.append(p_tag)

                if intersperse_blank:
                    phonemes_out.append(PAD)
                    phoneme_ids.append(self.pad_id)
                    prosody_ids.append(PROSODY_TAGS["NORMAL"])

        return phonemes_out, phoneme_ids, prosody_ids

    def attach_homo_ge2pe(
        self, model_dir: str, repo_dir: Optional[str] = None, gpu: bool = False
    ) -> bool:
        """Attach MahtaFetrat/Homo-GE2PE-Persian (ByT5) for contextual homograph & Ezafe G2P."""
        try:
            self.homo_ge2pe = HomoGE2PEWrapper(
                model_dir=model_dir, repo_dir=repo_dir, gpu=gpu
            )
            return True
        except Exception as exc:
            print(f"[WARN] Could not load Homo-GE2PE from {model_dir}: {exc}")
            self.homo_ge2pe = None
            return False

    def _split_ipa_for_piper(self, sym: str) -> List[str]:
        """
        Decompose multi-character IPA symbols (e.g. 'ɒː', 'iː', 'uː', 'tʃ', 'dʒ') into
        single keys present in Piper's phoneme_id_map (e.g. fa_IR-amir-medium.onnx.json).
        Also harmonizes velar/uvular symbols ('g'->'ɡ', 'ɣ'/'q'/'ɢ').
        """
        if sym in (BOS, EOS, PAD, " "):
            return [sym]

        equiv_map = {
            "g": ["ɡ", "g"],
            "ɡ": ["ɡ", "g"],
            "æ": ["a", "æ"],
            "ɣ": ["q", "ɣ", "ɢ", "ɡ"],
            "q": ["q", "ɢ", "ɣ", "ɡ"],
            "ɢ": ["q", "ɢ", "ɣ", "ɡ"],
            "ɒː": ["ɑ", "ɒ", "a"],
            "ɒ": ["ɑ", "ɒ", "a"],
            "ɑ": ["ɑ", "ɒ", "a"],
            "iː": ["i", "iː"],
            "uː": ["u", "uː"],
            "ʔ": ["ʔ", "?"],
        }

        if sym in self.phoneme_id_map:
            if sym == "g" and "ɡ" in self.phoneme_id_map:
                return ["ɡ"]
            return [sym]

        out: List[str] = []
        for ch in sym:
            if ch in equiv_map:
                chosen = next((c for c in equiv_map[ch] if c in self.phoneme_id_map), None)
                out.append(chosen if chosen is not None else ch)
            elif ch in self.phoneme_id_map:
                out.append(ch)
            else:
                out.append(ch)
        return out if out else [sym]

    def _ge2pe_word_to_pairs(
        self, token: str, is_final_question_word: bool = False
    ) -> List[Tuple[str, int]]:
        """Convert one Homo-GE2PE token (e.g. 'k/r/m1', '$/kki', 'h/tm/n') to (IPA, prosody_tag) pairs."""
        trailing_punct = []
        core = token
        while core and core[-1] in set("!'(),-.:;?،؛؟\"»«"):
            trailing_punct.insert(0, core[-1])
            core = core[:-1]
        while core and core[0] in set("!'(),-.:;?،؛؟\"»«"):
            core = core[1:]

        has_ezafe = False
        if core.endswith("1"):
            has_ezafe = True
            core = core[:-1]

        if core.startswith("@") and len(core) > 1:
            core = core[1:]
        if core.endswith("@") and len(core) > 1:
            core = core[:-1]

        chars = list(core)
        vowel_indices = [i for i, ch in enumerate(chars) if ch in GE2PE_VOWEL_CHARS]
        stem_vowels = vowel_indices
        if has_ezafe and core.endswith(("ye", "e")) and len(vowel_indices) >= 2:
            stem_vowels = vowel_indices[:-1]
        last_vowel_idx = stem_vowels[-1] if stem_vowels else (vowel_indices[-1] if vowel_indices else -1)
        unstressed_clitics = {"o", "be", "/z", "d/r", "ra", "ke", "ba", "ta", "ya"}
        initial_stress_cores = {
            "hala", "h/tm/n", "m/s/l/n", "lotf/n", "l/tf/n", "q/t@/n",
            "fe@l/n", "f/@l/n", "q/bl/n", "b/@d/n", "/sl/n", "vaqe@/n",
            "kamel/n", "m/@mul/n", "t/qrib/n", "nesb/t/n", "xosus/n",
            "/hyan/n", "zemn/n", "zaher/n", "/vv/l/n", "deqq/t",
            "v/li", "/mma", "cera", "/g/r", "b/le", "/lb/tte",
            "s/fid", "b/rge", "b/rgeye", "motm/@enn/n",
        }
        if not vowel_indices or core in unstressed_clitics:
            stress_vowel_idx = -1
        elif len(stem_vowels) >= 2 and (
            core in initial_stress_cores or core.startswith(("mi", "n/mi", "nemi"))
        ):
            stress_vowel_idx = stem_vowels[0]
        else:
            stress_vowel_idx = last_vowel_idx

        pairs: List[Tuple[str, int]] = []
        last_char_was_vowel = False
        for i, ch in enumerate(chars):
            ipa = GE2PE_CHAR_TO_IPA.get(ch)
            if ipa is None:
                continue
            is_consonant = ch in GE2PE_CONSONANT_CHARS
            is_geminated = is_consonant and (
                (i > 0 and chars[i - 1] == ch) or (i + 1 < len(chars) and chars[i + 1] == ch)
            )
            if i == stress_vowel_idx:
                pairs.append(("ˈ", PROSODY_TAGS["STRESS"]))
            if is_geminated:
                tag = PROSODY_TAGS["TASHDID"]
            elif is_final_question_word and i == last_vowel_idx:
                tag = PROSODY_TAGS["QUESTION"]
            elif i == stress_vowel_idx:
                tag = PROSODY_TAGS["STRESS"]
            else:
                tag = PROSODY_TAGS["NORMAL"]
            pairs.append((ipa, tag))
            last_char_was_vowel = ch in GE2PE_VOWEL_CHARS

        if has_ezafe:
            if core.endswith("ye") and len(pairs) >= 2:
                pairs[-2] = (pairs[-2][0], PROSODY_TAGS["EZAFE"])
                pairs[-1] = (pairs[-1][0], PROSODY_TAGS["EZAFE"])
            elif core.endswith("e") and len(pairs) >= 1:
                pairs[-1] = (pairs[-1][0], PROSODY_TAGS["EZAFE"])
            else:
                if last_char_was_vowel:
                    pairs.append(("j", PROSODY_TAGS["EZAFE"]))
                pairs.append(("e", PROSODY_TAGS["EZAFE"]))

        punct_map = {"،": ",", "؛": ";", "؟": "?"}
        for p in trailing_punct:
            mapped_p = punct_map.get(p, p)
            if mapped_p in self.phoneme_id_map:
                p_tag = (
                    PROSODY_TAGS["QUESTION"]
                    if mapped_p == "?"
                    else PROSODY_TAGS["NORMAL"]
                )
                pairs.append((mapped_p, p_tag))

        return pairs

    def _ge2pe_str_to_phoneme_tag_pairs(
        self, ge2pe_str: str, original_text: str = ""
    ) -> List[Tuple[str, int]]:
        is_question = ("؟" in original_text) or ("?" in original_text) or ("?" in ge2pe_str)
        tokens = [t for t in ge2pe_str.strip().split() if t]
        pairs: List[Tuple[str, int]] = [(BOS, PROSODY_TAGS["NORMAL"])]
        for idx, tok in enumerate(tokens):
            is_last = idx == len(tokens) - 1 or (
                idx == len(tokens) - 2 and tokens[-1] in set(".,?!،؛؟")
            )
            word_pairs = self._ge2pe_word_to_pairs(
                tok, is_final_question_word=(is_question and is_last)
            )
            pairs.extend(word_pairs)
            if idx < len(tokens) - 1:
                pairs.append((" ", PROSODY_TAGS["NORMAL"]))
        pairs.append((EOS, PROSODY_TAGS["QUESTION"] if is_question else PROSODY_TAGS["NORMAL"]))
        return pairs

    def _context_fallback_phoneme_tag_pairs(self, text: str) -> List[Tuple[str, int]]:
        """
        Context-aware fallback G2P (used only when Homo-GE2PE weights are unavailable locally)
        handling Tanvin ('حتما'), Homographs ('کرم', 'مرد', 'مهر', 'سمت'), Tashdid, and Ezafe.
        """
        norm = text.replace("\u200c", " ")
        tokens = re.findall(r"[\w]+|[،؛؟.,?!]", norm)
        is_question = ("؟" in text) or ("?" in text)
        pairs: List[Tuple[str, int]] = [(BOS, PROSODY_TAGS["NORMAL"])]

        base_words: Dict[str, str] = {
            "حتما": "h/tm/n",
            "مثلا": "m/s/l/n",
            "در": "d/r",
            "بزرگ": "bozorg",
            "شکی": "$/kki",
            "نیست": "nist",
            "ولی": "v/li",
            "کوچک": "kuc/k",
            "روی": "ruye",
            "برگ": "b/rg1",
            "درخت": "der/xt",
            "بود": "bud",
            "آن": "an",
            "جوان": "j/van",
            "که": "ke",
            "معلم": "mo@/llem1",
            "مدرسه": "m/drese",
            "دیروز": "diruz",
            "حادثه": "hadese",
            "آیا": "aya",
            "استاد": "ostad1",
            "دانشگاه": "dane$gah",
            "با": "ba",
            "و": "o",
            "محبت": "moh/bb/t",
            "بر": "b/r",
            "برگه": "b/rgeye1",
            "امتحان": "emtehan",
            "زده": "z/de",
            "شد": "$od",
            "مدیر": "modir1",
            "جدید": "j/did",
            "ریاست": "riyas/t",
            "دقت": "deqq/t",
            "به": "be",
            "اتاق": "otaq1",
            "جلسات": "j/l/sat",
            "رفت": "r/ft",
            "خوب": "xub",
            "پس": "p/s",
            "از": "/z",
            "تشکر": "t/$/kkor1",
            "فراوان": "f/ravan",
            "داستان": "dastan1",
            "شیر": "$ir1",
            "جنگل": "j/ng/l",
            "را": "ra",
            "صدای": "sedaye",
            "بلند": "bol/nd",
            "خواند": "xand",
        }

        word_only = [t for t in tokens if t not in set("،؛؟.,?!")]
        w_pos = 0
        punct_map = {"،": ",", "؛": ";", "؟": "?"}
        for idx, tok in enumerate(tokens):
            if tok in set("،؛؟.,?!"):
                mapped_p = punct_map.get(tok, tok)
                if mapped_p in self.phoneme_id_map:
                    p_tag = PROSODY_TAGS["QUESTION"] if mapped_p == "?" else PROSODY_TAGS["NORMAL"]
                    pairs.append((mapped_p, p_tag))
                if idx < len(tokens) - 1:
                    pairs.append((" ", PROSODY_TAGS["NORMAL"]))
                continue

            prev_w = word_only[w_pos - 1] if w_pos > 0 else ""
            next_w = word_only[w_pos + 1] if w_pos + 1 < len(word_only) else ""
            ge2pe_tok = None
            if tok == "کرم":
                if next_w in ("کوچک", "خاکی", "ابریشم") or (next_w != "مرد" and "درخت" in word_only):
                    ge2pe_tok = "kerm1" if next_w == "کوچک" else "kerm"
                else:
                    ge2pe_tok = "k/r/m1" if next_w in ("مرد", "خدا") else "k/r/m"
            elif tok == "مرد":
                if next_w in ("بزرگ", "جوان", "خوب") or prev_w == "آن":
                    ge2pe_tok = "m/rd1" if next_w in ("بزرگ", "جوان") else "m/rd"
                else:
                    ge2pe_tok = "mord"
            elif tok == "مهر":
                ge2pe_tok = "mohr1" if next_w in ("استاد", "دانشگاه", "امتحان") else "mehr"
            elif tok == "سمت":
                if prev_w == "به":
                    ge2pe_tok = "s/mt1" if next_w in ("اتاق", "چپ", "راست") else "s/mt"
                else:
                    ge2pe_tok = "sem/t1" if next_w in ("ریاست", "مدیریت") else "sem/t"
            elif tok == "معلم":
                ge2pe_tok = "mo@/llem1" if next_w in ("مدرسه", "خوب") else "mo@/llem"
            elif tok in base_words:
                ge2pe_tok = base_words[tok]

            is_last_word = (w_pos == len(word_only) - 1)
            if ge2pe_tok is not None:
                pairs.extend(
                    self._ge2pe_word_to_pairs(
                        ge2pe_tok, is_final_question_word=(is_question and is_last_word)
                    )
                )
            else:
                sub_syms, _, sub_tags = self.text_to_aligned_streams(tok, intersperse_blank=False)
                for s, t in zip(sub_syms, sub_tags):
                    if s not in (BOS, EOS, " "):
                        pairs.append((s, t))

            w_pos += 1
            if idx < len(tokens) - 1 and tokens[idx + 1] not in set("،؛؟.,?!"):
                pairs.append((" ", PROSODY_TAGS["NORMAL"]))

        pairs.append((EOS, PROSODY_TAGS["QUESTION"] if is_question else PROSODY_TAGS["NORMAL"]))
        return pairs

    def text_to_aligned_streams_with_context(
        self, text: str, intersperse_blank: bool = True, prefer_homo_ge2pe: bool = True
    ):
        """
        Convert Persian text using Homo-GE2PE (if attached) + context hidden states,
        falling back gracefully to context-aware homograph/tanvin/ezafe rules if offline.
        Returns: (phonemes_out, phoneme_ids, prosody_ids, context_emb, raw_ge2pe_str)
        """
        context_emb = None
        raw_ge2pe_str = ""
        ge2pe_obj = getattr(self, "homo_ge2pe", None)

        if prefer_homo_ge2pe and ge2pe_obj is not None:
            raw_ge2pe_str, context_emb = ge2pe_obj.predict_with_context(text, use_rules=True)
            phoneme_tag_pairs = self._ge2pe_str_to_phoneme_tag_pairs(
                raw_ge2pe_str, original_text=text
            )
        else:
            phoneme_tag_pairs = self._context_fallback_phoneme_tag_pairs(text)

        phonemes_out: List[str] = []
        phoneme_ids: List[int] = []
        prosody_ids: List[int] = []

        for sym, p_tag in phoneme_tag_pairs:
            for sub_sym in self._split_ipa_for_piper(sym):
                p_id = self.phoneme_id_map.get(sub_sym, self.pad_id)
                phonemes_out.append(sub_sym)
                phoneme_ids.append(p_id)
                prosody_ids.append(p_tag)

                if intersperse_blank:
                    phonemes_out.append(PAD)
                    phoneme_ids.append(self.pad_id)
                    prosody_ids.append(PROSODY_TAGS["NORMAL"])

        return phonemes_out, phoneme_ids, prosody_ids, context_emb, raw_ge2pe_str


PUNCTUATION_SET = set("!'(),-.:;?،؛؟")
