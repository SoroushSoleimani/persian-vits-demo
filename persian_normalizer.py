"""Stage 1: Persian Text Normalization for VITS + Piper TTS Pipeline.

Handles:
  - Arabic -> Persian character unification (ك->ک, ي/ى->ی, ة->ه, أ/إ->ا, ؤ->و)
  - ZWNJ (\u200c) preservation inside Persian morphology (می‌رود, کتاب‌ها)
  - Numbers: integers, decimals, percentages (%), and Persian/English digits
  - Jalali/numeric dates (e.g., ۱۴۰۳/۰۸/۲۵ -> بیست و پنجم آبان هزار و چهارصد و سه)
  - Common Persian abbreviations (د., م., ه.ش, ه.ق, ص., ش., ج., km, kg)
"""

import re
from typing import Dict, List

ZWNJ = "\u200c"

_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

_ONES = [
    "", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه",
    "ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده",
]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = [
    "", "صد", "دویست", "سیصد", "چهارصد", "پانصد",
    "ششصد", "هفتصد", "هشتصد", "نهصد",
]
_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]

_JALALI_MONTHS = {
    1: "فروردین",
    2: "اردیبهشت",
    3: "خرداد",
    4: "تیر",
    5: "مرداد",
    6: "شهریور",
    7: "مهر",
    8: "آبان",
    9: "آذر",
    10: "دی",
    11: "بهمن",
    12: "اسفند",
}

_DAY_ORDINALS = {
    1: "یکم", 2: "دوم", 3: "سوم", 4: "چهارم", 5: "پنجم",
    6: "ششم", 7: "هفتم", 8: "هشتم", 9: "نهم", 10: "دهم",
    11: "یازدهم", 12: "دوازدهم", 13: "سیزدهم", 14: "چهاردهم", 15: "پانزدهم",
    16: "شانزدهم", 17: "هفدهم", 18: "هجدهم", 19: "نوزدهم", 20: "بیستم",
    21: "بیست و یکم", 22: "بیست و دوم", 23: "بیست و سوم", 24: "بیست و چهارم", 25: "بیست و پنجم",
    26: "بیست و ششم", 27: "بیست و هفتم", 28: "بیست و هشتم", 29: "بیست و نهم", 30: "سی‌ام",
    31: "سی و یکم",
}

_ABBREVIATIONS: List[tuple] = [
    (r"(?<!\S)د\.(?=\s|$)", "دکتر"),
    (r"(?<!\S)م\.(?=\s|$)", "میلادی"),
    (r"(?<!\S)ه\.ش\.?(?=\s|$)", "هجری شمسی"),
    (r"(?<!\S)ه\.ق\.?(?=\s|$)", "هجری قمری"),
    (r"(?<!\S)ص\.(?=\s|$|\d)", "صفحه"),
    (r"(?<!\S)ش\.(?=\s|$|\d)", "شماره"),
    (r"(?<!\S)ج\.(?=\s|$|\d)", "جلد"),
    (r"(?<!\S)ت\.(?=\s|$)", "تومان"),
    (r"\bkm\b", "کیلومتر"),
    (r"\bkg\b", "کیلوگرم"),
    (r"\bcm\b", "سانتی‌متر"),
    (r"\bmm\b", "میلی‌متر"),
]


def _three_digits_to_persian(n: int) -> str:
    parts: List[str] = []
    h = n // 100
    rem = n % 100
    if h:
        parts.append(_HUNDREDS[h])
    if rem:
        if rem < 20:
            parts.append(_ONES[rem])
        else:
            t = rem // 10
            o = rem % 10
            if o:
                parts.append(f"{_TENS[t]} و {_ONES[o]}")
            else:
                parts.append(_TENS[t])
    return " و ".join(parts)


def int_to_persian_words(n: int) -> str:
    """Convert non-negative or negative integer into spoken Persian words."""
    if n == 0:
        return "صفر"
    if n < 0:
        return "منفی " + int_to_persian_words(-n)

    chunks: List[str] = []
    scale_idx = 0
    while n > 0:
        rem = n % 1000
        if rem:
            words = _three_digits_to_persian(rem)
            scale = _SCALES[scale_idx] if scale_idx < len(_SCALES) else ""
            chunks.append(f"{words} {scale}".strip() if scale else words)
        n //= 1000
        scale_idx += 1
    return " و ".join(reversed(chunks))


def normalize_arabic_to_persian(text: str) -> str:
    """Unify Arabic variants and digits into clean Persian orthography while keeping ZWNJ."""
    t = text.translate(_PERSIAN_DIGITS)
    replacements: Dict[str, str] = {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "ة": "ه",
        "أ": "ا",
        "إ": "ا",
        "ؤ": "و",
        "ۀ": "ه",
        "٬": ",",
        "٫": ".",
    }
    for src, dst in replacements.items():
        t = t.replace(src, dst)
    # Collapse spaces around ZWNJ without deleting ZWNJ inside compound words
    t = re.sub(rf"\s*{ZWNJ}\s*", ZWNJ, t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t


def normalize_dates(text: str) -> str:
    """Convert YYYY/MM/DD or YYYY-MM-DD dates into spoken Persian."""
    def _repl(m: re.Match) -> str:
        y = int(m.group(1))
        mo = int(m.group(2))
        d = int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        d_str = _DAY_ORDINALS.get(d, int_to_persian_words(d))
        m_str = _JALALI_MONTHS[mo]
        y_str = int_to_persian_words(y)
        return f"{d_str} {m_str} {y_str}"

    return re.sub(r"\b(1[234]\d{2}|20\d{2})[/\-](0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])\b", _repl, text)


def normalize_abbreviations(text: str) -> str:
    """Expand common Persian & unit abbreviations into spoken words."""
    out = text
    for pattern, repl in _ABBREVIATIONS:
        out = re.sub(pattern, repl, out)
    return out


def normalize_numbers(text: str) -> str:
    """Convert percentages, decimals, and integers into spoken Persian words."""
    def _pct_repl(m: re.Match) -> str:
        num_str = m.group(1)
        if "." in num_str:
            whole, frac = num_str.split(".", 1)
            return f"{int_to_persian_words(int(whole))} ممیز {int_to_persian_words(int(frac))} درصد"
        return f"{int_to_persian_words(int(num_str))} درصد"

    out = re.sub(r"\b(\d+(?:\.\d+)?)\s*[%٪]", _pct_repl, text)

    def _dec_repl(m: re.Match) -> str:
        whole, frac = m.group(1), m.group(2)
        return f"{int_to_persian_words(int(whole))} ممیز {int_to_persian_words(int(frac))}"

    out = re.sub(r"\b(\d+)\.(\d+)\b", _dec_repl, out)

    def _int_repl(m: re.Match) -> str:
        return int_to_persian_words(int(m.group(0)))

    out = re.sub(r"\b\d+\b", _int_repl, out)
    return re.sub(r"[ \t]+", " ", out).strip()


def normalize_persian_text(text: str) -> str:
    """Full Stage 1 normalization pipeline: chars -> dates -> abbreviations -> numbers -> ZWNJ-safe spacing."""
    t = normalize_arabic_to_persian(text)
    t = normalize_dates(t)
    t = normalize_abbreviations(t)
    t = normalize_numbers(t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t
