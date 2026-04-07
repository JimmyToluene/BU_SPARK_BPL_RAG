"""
Shared OCR quality metrics used by ocr_eval, compare_tiers, and EDA scripts.
"""

import re

from wordfreq import zipf_frequency

from .constants import (
    HISTORICAL_WHITELIST,
    OCR_JUNK_CHARS_RE,
    PUNCT_ONLY,
    REAL_SHORT_WORDS,
)


def character_anomaly_rate(text: str) -> float:
    """Fraction of characters that are known OCR junk symbols."""
    if not text:
        return 0.0
    return round(len(OCR_JUNK_CHARS_RE.findall(text)) / len(text), 4)


def non_alpha_token_rate(text: str) -> float:
    """Fraction of non-punctuation tokens that contain no letters."""
    tokens = text.split()
    if not tokens:
        return 0.0
    candidates = [t for t in tokens if not PUNCT_ONLY.match(t)]
    if not candidates:
        return 0.0
    non_alpha = sum(1 for t in candidates if not re.search(r"[a-zA-Z]", t))
    return round(non_alpha / len(candidates), 4)


def short_token_rate(text: str) -> float:
    """Fraction of alpha tokens that reduce to <=2 letters (excluding real short words)."""
    tokens = [t for t in text.split() if re.search(r"[a-zA-Z]", t)]
    if not tokens:
        return 0.0
    short = sum(
        1 for t in tokens
        if len(re.sub(r"[^a-zA-Z]", "", t)) <= 2
        and re.sub(r"[^a-zA-Z]", "", t).lower() not in REAL_SHORT_WORDS
    )
    return round(short / len(tokens), 4)


def oov_rate(text: str, wordlist: set = None) -> float:
    """Fraction of tokens not in the reference wordlist."""
    words = text.split()
    if not words:
        return 0.0

    reference = HISTORICAL_WHITELIST.copy()
    if wordlist:
        reference.update(wordlist)

    def is_known(word: str) -> bool:
        cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", word).lower()
        if not cleaned or len(cleaned) <= 2:
            return True
        if cleaned in reference:
            return True
        if cleaned[0].isupper():
            return True
        return zipf_frequency(cleaned, "en") > 1.5

    oov = sum(1 for w in words if not is_known(w))
    return round(oov / len(words), 4)


def composite_score(
    char_anomaly: float,
    non_alpha: float,
    short_tok: float,
) -> float:
    """Single 0-1 quality score. Higher = cleaner."""
    noise = 0.40 * char_anomaly + 0.35 * non_alpha + 0.25 * short_tok
    return round(max(0.0, min(1.0, 1.0 - noise)), 4)
