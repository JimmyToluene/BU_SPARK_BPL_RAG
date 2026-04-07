"""
Shared constants used across the pipeline and analysis scripts.
"""

import re

# ── Word extraction ──────────────────────────────────────────────────────────
WORD_RE = re.compile(r"[A-Za-z]+")

# ── OCR junk detection ───────────────────────────────────────────────────────
OCR_JUNK_CHARS_SET = set("|¦©°∫¬§±×÷¤¨¸˜")
OCR_JUNK_CHARS_RE = re.compile(r"[|¦©°∫¬§±×÷¤¨¸˜]")

# ── Token classification ────────────────────────────────────────────────────
PUNCT_ONLY = re.compile(r"^[.,;:!?\-\"'()]+$")
REAL_SHORT_WORDS = {
    "a", "i", "o",
    "an", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "me", "my", "no", "of", "on", "or",
    "so", "to", "up", "us", "we",
    "mr", "dr", "st", "vs",
}

# ── Historical vocabulary whitelist ──────────────────────────────────────────
HISTORICAL_WHITELIST = {
    "ye", "thee", "thou", "thy", "thine",
    "hath", "doth", "hast", "wilt", "shalt",
    "mayst", "wouldst", "couldst", "shouldst", "mightst",
    "art", "wert", "hadst",
    "hither", "thither", "whither", "wherein", "whereof",
    "thereof", "hereof", "hereby", "therein",
    "methinks", "perchance", "haply", "belike", "forsooth",
    "nay", "aye", "yea", "verily", "prithee", "pray",
    "forthwith", "heretofore", "henceforth", "notwithstanding",
    "publick", "musick", "logick", "ethick", "rhetorick",
    "connexion", "inflexion", "reflexion", "compleat",
    "shew", "shewed", "shewn", "staid", "spake", "writ",
    "honour", "colour", "labour", "neighbour", "favour",
    "centre", "theatre", "fibre", "lustre", "sceptre",
    "recognise", "organise", "realise", "civilise",
    "defence", "offence", "pretence", "licence",
    "esq", "viz", "ibid", "idem", "supra", "infra",
    "aforesaid", "aforementioned", "heretofore",
    "whereas", "whereby", "whereupon", "whereunto",
    "mr", "mrs", "dr", "sr", "jr", "rev", "hon", "gen",
}
