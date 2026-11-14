"""
Persian IPA Symbols & Prosody Feature Definitions for Custom Piper/VITS Fine-Tuning.
Compatible with pre-trained Piper Persian (fa_IR) phoneme ID maps.
"""

from typing import Dict, List

# Standard VITS / Piper special tokens
PAD = "_"
BOS = "^"
EOS = "$"
SPACE = " "

# Punctuation & intonation symbols
PUNCTUATION = list("!'(),-.:;?،؛؟ ")

# Core Persian IPA vowels & consonants (compatible with espeak-ng fa & Persian-G2P/Hezar)
PERSIAN_IPA = [
    # Vowels
    "a", "æ", "e", "i", "iː", "o", "u", "uː", "ɑ", "ɒ", "ɒː",
    # Consonants
    "b", "d", "f", "g", "ɡ", "h", "j", "k", "l", "m", "n",
    "p", "q", "1", "r", "s", "t", "v", "w", "x", "z",
    "ʃ", "ʒ", "tʃ", "dʒ", "ʔ", "ɣ", "ŋ",
    # Length & stress modifiers in IPA
    "ˈ", "ˌ", "ː"
]

# Prosody / Phonetic Feature Tags (Upgrade #1 parallel input stream)
PROSODY_TAGS: Dict[str, int] = {
    "NORMAL": 0,      # Standard phoneme (Zero-initialized so base checkpoint is unaffected)
    "STRESS": 1,      # Primary lexical stress (تکیه کلام)
    "TASHDID": 2,     # Geminated consonant (تشدید - e.g., معلّم)
    "EZAFE": 3,       # Persian Ezafe connector (-e / کسره اضافه)
    "QUESTION": 4,    # Rising question intonation (؟)
}

NUM_PROSODY_TAGS = len(PROSODY_TAGS)


def build_default_symbol_map() -> Dict[str, int]:
    """Builds a default phoneme-to-ID map if no base model.onnx.json is loaded yet."""
    all_symbols: List[str] = [PAD, BOS, EOS] + PUNCTUATION + PERSIAN_IPA
    unique_symbols: List[str] = []
    for sym in all_symbols:
        if sym not in unique_symbols:
            unique_symbols.append(sym)
    return {sym: idx for idx, sym in enumerate(unique_symbols)}
