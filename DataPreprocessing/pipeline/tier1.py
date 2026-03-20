import re

REPLACEMENTS = {
    '|':  'I',
    '¦':  'I',
    '©':  'o',
    '°':  'o',
    'ſ':  's',
    '∫':  's',
    '\x00': '',
}

def tier1_clean(text: str) -> str:
    # 1. Character replacements before stripping
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)

    # 2. Rejoin hyphenated line breaks
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 3. Preserve paragraph breaks, collapse other whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    # 4. Fix punctuation spacing
    text = re.sub(r"\s([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # 5. Strip remaining junk after replacements
    text = re.sub(r"[^\w\s.,!?;:'\"\-\u2013\u2014()]", "", text)

    return text.strip()


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=(
            "Run Tier 1 OCR cleaning on a JSON file produced by Phase 2/3.\n"
            "Adds a new field `tier_1_cleaned` to each record.\n"
            "By default it does NOT overwrite the original text field.\n"
            "Use `--overwrite_raw_text` if you want the input field replaced in-place."
        )
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to input JSON file (e.g. .../1900.json or .../1940-03-15.json).",
    )
    parser.add_argument(
        "--field",
        default="raw_text",
        help="Record field to clean (default: raw_text).",
    )
    parser.add_argument(
        "--new_field",
        default="tier_1_cleaned",
        help="Field to store cleaned text (default: tier_1_cleaned).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Optional output file path. If omitted, writes back to the input file (in-place)."
        ),
    )
    parser.add_argument(
        "--overwrite_raw_text",
        action="store_true",
        help="Overwrite record[--field] with the cleaned value (in-place).",
    )

    args = parser.parse_args()

    in_path = Path(args.path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    out_path = Path(args.out) if args.out else in_path

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        raise SystemExit(f"Unsupported JSON structure: {type(data).__name__}")

    if not isinstance(records, list):
        raise SystemExit("Expected 'records' to be a list (or top-level to be a list).")

    cleaned_count = 0
    skipped_missing_field = 0

    for rec in records:
        if not isinstance(rec, dict):
            continue
        raw = rec.get(args.field, "")
        if not raw:
            skipped_missing_field += 1
            continue
        cleaned = tier1_clean(raw)
        rec[args.new_field] = cleaned
        if args.overwrite_raw_text:
            rec[args.field] = cleaned
        cleaned_count += 1

    # Write output with the same top-level structure we read
    if isinstance(data, dict):
        data["records"] = records
        output_data = data
    else:
        output_data = records

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Tier 1 cleaned {cleaned_count} record(s).")
    if skipped_missing_field:
        print(f"Skipped {skipped_missing_field} record(s) due to missing/empty '{args.field}'.")
    print(f"Saved: {out_path}")
