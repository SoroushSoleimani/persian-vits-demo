"""
Step 1: Persian Text Normalization (Hazm / Parsivar optional + deterministic rules).

Handles:
  - Arabic -> Persian character normalization
  - ZWNJ (\u200c) normalization (preserve or space-normalize)
  - Common Persian abbreviations (ه.ش, ه.ق, ق.م, دکتر, درصد, ...)
  - Persian/Shamsi dates (e.g. 1403/08/15 -> پانزدهم آبان هزار و چهارصد و سه)
  - Integers, negatives, and decimal numbers -> spoken Persian words
"""

import re
from typing import Optional

_ARABIC_TO_PERSIAN = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ۀ": "ه",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٫": ".",
        "٪": "%",
    }
)

_ONES = [
    "صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه",
    "ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده",
]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = [
    "", "صد", "دویست", "سیصد", "چهارصد", "پانصد",
    "ششصد", "هفتصد", "هشتصد", "نهصد",
]
_SCALES = ["", "هزار", "میلیون", "میلیارد", "تریلیون"]

_SHAMSI_MONTHS = {
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

_ABBREVIATIONS = [
    (re.compile(r"(?<!\S)ق\.?\s*م\.?(?!\S)"), "قبل از میلاد"),
    (re.compile(r"(?<!\S)ه\.?\s*ش\.?(?!\S)"), "هجری شمسی"),
    (re.compile(r"(?<!\S)ه\.?\s*ق\.?(?!\S)"), "هجری قمری"),
    (re.compile(r"(?<!\S)ص\.?\s*ب\.?(?!\S)"), "صندوق پستی"),
    (re.compile(r"(?<!\S)ج\.?\s*ا\.?\s*ا\.?(?!\S)"), "جمهوری اسلامی ایران"),
    (re.compile(r"(?<!\S)د\.(?=\s+\S)"), "دکتر"),
    (re.compile(r"(?<!\S)ص\.(?=\s*\d)"), "صفحه"),
    (re.compile(r"(?<!\S)ج\.(?=\s*\d)"), "جلد"),
    (re.compile(r"(?<!\S)ش\.(?=\s*\d)"), "شماره"),
    (re.compile(r"(?<!\S)ت\.(?=\s*\d)"), "تلفن"),
    (re.compile(r"\bkm\b", re.IGNORECASE), "کیلومتر"),
    (re.compile(r"\bkg\b", re.IGNORECASE), "کیلوگرم"),
    (re.compile(r"\bcm\b", re.IGNORECASE), "سانتی‌متر"),
    (re.compile(r"\bmm\b", re.IGNORECASE), "میلی‌متر"),
    (re.compile(r"\bm\b"), "متر"),
]

_OPTIONAL_EXTERNAL_NORMALIZER = None
_EXTERNAL_CHECKED = False


def _get_external_normalizer():
    global _OPTIONAL_EXTERNAL_NORMALIZER, _EXTERNAL_CHECKED
    if _EXTERNAL_CHECKED:
        return _OPTIONAL_EXTERNAL_NORMALIZER
    _EXTERNAL_CHECKED = True
    try:
        from hazm import Normalizer as HazmNormalizer  # type: ignore

        _OPTIONAL_EXTERNAL_NORMALIZER = HazmNormalizer()
        return _OPTIONAL_EXTERNAL_NORMALIZER
    except Exception:
        pass
    try:
        from parsivar import Normalizer as ParsivarNormalizer  # type: ignore

        _OPTIONAL_EXTERNAL_NORMALIZER = ParsivarNormalizer()
        return _OPTIONAL_EXTERNAL_NORMALIZER
    except Exception:
        pass
    return None


def _three_digits_to_words(n: int) -> str:
    parts = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if rem:
        if rem < 20:
            parts.append(_ONES[rem])
        else:
            t, o = divmod(rem, 10)
            parts.append(_TENS[t])
            if o:
                parts.append(_ONES[o])
    return " و ".join(parts)


def int_to_persian_words(n: int) -> str:
    """Convert an integer to spoken Persian words."""
    if n == 0:
        return _ONES[0]
    if n < 0:
        return "منفی " + int_to_persian_words(-n)

    groups = []
    scale_idx = 0
    while n > 0:
        n, chunk = divmod(n, 1000)
        if chunk:
            if scale_idx == 1 and chunk == 1:
                words = _SCALES[scale_idx]
            else:
                words = _three_digits_to_words(chunk)
                if scale_idx < len(_SCALES) and _SCALES[scale_idx]:
                    words = f"{words} {_SCALES[scale_idx]}"
            groups.append(words)
        scale_idx += 1
    return " و ".join(reversed(groups))


def int_to_persian_ordinal(n: int) -> str:
    """Convert a positive integer (e.g. day of month) to Persian ordinal words."""
    if n == 1:
        return "اول"
    if n == 3:
        return "سوم"
    if n == 30:
        return "سی‌ام"
    base = int_to_persian_words(n)
    if base.endswith("سه"):
        return base[:-2] + "سوم"
    return base + "م"


def _expand_date_match(match: re.Match) -> str:
    y_str, m_str, d_str = match.group(1), match.group(2), match.group(3)
    y, m, d = int(y_str), int(m_str), int(d_str)
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return f"{int_to_persian_words(y)} {int_to_persian_words(m)} {int_to_persian_words(d)}"
    day_w = int_to_persian_ordinal(d)
    month_w = _SHAMSI_MONTHS[m]
    year_w = int_to_persian_words(y)
    return f"{day_w} {month_w} {year_w}"


def _expand_number_match(match: re.Match) -> str:
    sign = match.group(1) or ""
    int_part = match.group(2)
    dec_part = match.group(3)
    pct = match.group(4) or ""

    prefix = "منفی " if sign == "-" else ""
    words = prefix + int_to_persian_words(int(int_part))
    if dec_part is not None:
        dec_stripped = dec_part.lstrip("0")
        if not dec_stripped:
            dec_words = "صفر"
        elif len(dec_part) > len(dec_stripped):
            dec_words = " ".join(_ONES[int(ch)] for ch in dec_part)
        else:
            dec_words = int_to_persian_words(int(dec_part))
        words = f"{words} ممیز {dec_words}"
    if pct == "%":
        words = f"{words} درصد"
    return words


def normalize_persian_text(
    text: str,
    *,
    use_external: bool = True,
    preserve_zwnj: bool = True,
    expand_numbers_and_dates: bool = True,
    normalize_question_mark: bool = True,
) -> str:
    """
    Full Persian text normalization (Stage 1 of the pipeline):
      1. Optional Hazm/Parsivar normalizer (if installed and use_external=True)
      2. Arabic -> Persian char & digit mapping
      3. Abbreviation expansion
      4. Date expansion (YYYY/MM/DD or YYYY-MM-DD)
      5. Number & percentage expansion
      6. ZWNJ and whitespace normalization
    """
    if not text:
        return ""

    t = text.strip()
    if use_external:
        ext = _get_external_normalizer()
        if ext is not None:
            try:
                t = ext.normalize(t)
            except Exception:
                pass

    t = t.translate(_ARABIC_TO_PERSIAN)
    if normalize_question_mark:
        t = t.replace("?", "؟")

    # Normalize ZWNJ runs and spaces around ZWNJ
    t = re.sub(r"\u200c+", "\u200c", t)
    t = re.sub(r"\s*\u200c\s*", "\u200c" if preserve_zwnj else " ", t)

    if expand_numbers_and_dates:
        # Abbreviations first (e.g. ص. 12 -> صفحه 12, ه.ش -> هجری شمسی)
        for pattern, repl in _ABBREVIATIONS:
            t = pattern.sub(repl, t)

        # Dates: YYYY/MM/DD or YYYY-MM-DD
        t = re.sub(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", _expand_date_match, t)

        # Numbers (optional leading '-', integer or decimal, optional '%')
        t = re.sub(r"(?<![\w.])(-?)(\d+)(?:\.(\d+))?(\s*%?)(?!\w)", _expand_number_match, t)
        t = t.replace("%", " درصد")

    if not preserve_zwnj:
        t = t.replace("\u200c", " ")

    t = re.sub(r"[ \t]+", " ", t).strip()
    return t
