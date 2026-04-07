import re
from spellchecker import SpellChecker
from wordfreq import zipf_frequency

from .constants import HISTORICAL_WHITELIST
from .tier1 import tier1_clean
from .utils import build_bln600_wordset


def tier2_clean(
    text: str,
    wordlist: set = None,
    spell_distance: int = 1,
    extra_whitelist: set = None,
    use_modern_dictionary: bool = True,
    modern_zipf_threshold: float = 0.0,
    run_tier1: bool = True,
) -> str:
    """
    Tier 2 cleaning — optional Tier 1 normalization + dictionary-based correction.

    Args:
        text:            Input text. If `run_tier1` is True, applies `tier1_clean()` first.
        wordlist:        Historical wordlist from build_bln600_wordlist().
                         Falls back to HISTORICAL_WHITELIST if None.
        spell_distance:  Levenshtein distance for corrections (1 = conservative, 2 = aggressive).
        extra_whitelist: Additional words to protect from correction.
        run_tier1:       If True, runs `tier1_clean()` on the input text first.
    """
    # Run Tier 1 first only if requested.
    if run_tier1:
        text = tier1_clean(text)

    # Build active whitelist
    whitelist = HISTORICAL_WHITELIST.copy()
    if wordlist:
        whitelist.update(wordlist)
    if extra_whitelist:
        whitelist.update(w.lower() for w in extra_whitelist)

    spell = SpellChecker(distance=spell_distance)
    spell.word_frequency.load_words(list(whitelist))

    modern_cache: dict[str, bool] = {}

    def is_modern_word(lower_core: str) -> bool:
        """
        Modern dictionary existence check using `wordfreq`.
        Returns True iff `zipf_frequency(word, 'en')` is above `modern_zipf_threshold`.
        """
        if lower_core in modern_cache:
            return modern_cache[lower_core]
        freq = zipf_frequency(lower_core, "en")
        ok = freq > modern_zipf_threshold
        modern_cache[lower_core] = ok
        return ok

    def correct_word(word: str) -> str:
        prefix = re.match(r"^([^a-zA-Z]*)", word).group(1)
        suffix = re.search(r"([^a-zA-Z]*)$", word).group(1)
        core = word[len(prefix): len(word) - len(suffix)] if suffix else word[len(prefix):]

        if not core:
            return word

        lower = core.lower()

        # If this word exists in modern English, don't "correct" it.
        # This matches the desired rule: valid if (modern exists) OR (BLN600 exists).
        if use_modern_dictionary and len(lower) > 2 and is_modern_word(lower):
            return word

        if (lower in whitelist
                or core[0].isupper()
                or re.search(r"\d", core)
                or len(core) <= 2):
            return word

        correction = spell.correction(lower)
        if correction and correction != lower:
            if core.isupper():
                correction = correction.upper()
            elif core[0].isupper():
                correction = correction.capitalize()
            return prefix + correction + suffix

        return word

    tokens = re.split(r"(\s+)", text)
    corrected = [correct_word(t) if not re.match(r"^\s+$", t) else t for t in tokens]
    return "".join(corrected)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(
        description=(
            "Run Tier 2 cleaning on a JSON file.\n"
            "Adds a new field (default: tier_2_cleaned) and optionally overwrites "
            "the input field.\n"
            "By default, it expects the input text to already be Tier 1 cleaned "
            "(so it will NOT call tier1_clean)."
        )
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to input JSON file (e.g. .../1900.json).",
    )
    parser.add_argument(
        "--field",
        default="tier_1_cleaned",
        help="Record field to clean (default: tier_1_cleaned).",
    )
    parser.add_argument(
        "--new_field",
        default="tier_2_cleaned",
        help="Field name to write Tier 2 output (default: tier_2_cleaned).",
    )
    parser.add_argument(
        "--bln600_root",
        default="DataPreprocessing/BLN600",
        help="Path to DataPreprocessing/BLN600 (default).",
    )
    parser.add_argument(
        "--spell_distance",
        type=int,
        default=1,
        help="Spell correction edit distance (default: 1).",
    )
    parser.add_argument(
        "--modern_zipf_threshold",
        type=float,
        default=0.0,
        help="Modern existence threshold using wordfreq zipf_frequency (default: 0.0).",
    )
    parser.add_argument(
        "--use_modern_dictionary",
        action="store_true",
        help="Enable modern-dictionary protection (default: enabled).",
    )
    parser.add_argument(
        "--no_use_modern_dictionary",
        action="store_true",
        help="Disable modern-dictionary protection.",
    )
    parser.add_argument(
        "--run_tier1",
        action="store_true",
        help="Actually run tier1_clean() inside Tier 2 (not recommended if your input is already Tier 1 cleaned).",
    )
    parser.add_argument(
        "--overwrite_input_field",
        action="store_true",
        help="Also overwrite the input --field with the Tier 2 cleaned output.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output file path. If omitted, modifies the input file in-place.",
    )
    args = parser.parse_args()

    in_path = _Path(args.path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    out_path = _Path(args.out) if args.out else in_path

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
        data = {"records": records}
    else:
        raise SystemExit(f"Unsupported JSON structure: top-level is {type(data).__name__}")

    if not isinstance(records, list):
        raise SystemExit("Expected records to be a list.")

    bln_words = build_bln600_wordlist(args.bln600_root)

    use_modern = True
    if args.no_use_modern_dictionary:
        use_modern = False
    if args.use_modern_dictionary:
        use_modern = True

    cleaned_count = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        text = rec.get(args.field, "")
        if not text:
            continue
        cleaned = tier2_clean(
            text,
            wordlist=bln_words,
            spell_distance=args.spell_distance,
            use_modern_dictionary=use_modern,
            modern_zipf_threshold=args.modern_zipf_threshold,
            run_tier1=args.run_tier1,
        )
        rec[args.new_field] = cleaned
        if args.overwrite_input_field:
            rec[args.field] = cleaned
        cleaned_count += 1

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Tier 2 cleaned {cleaned_count} record(s).")
    print(f"Saved: {out_path}")
