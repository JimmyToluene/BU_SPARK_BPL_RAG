"""
Split bpl_checkpoint.json into per-year files.

Output: bpl_<YEAR>.json for each year found in the data.
Each file has the same structure as the checkpoint:
  { "records": [...], "processed_ids": [...], "count": N }

Usage:
  python split_checkpoint.py
  python split_checkpoint.py --input bpl_checkpoint.json --outdir chunks
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def extract_year(record: dict) -> str:
    """Extract year from a record, trying multiple fields."""
    # Try year field first (e.g. [1900])
    year_list = record.get("year", [])
    if year_list:
        return str(year_list[0])

    # Try date_start (e.g. "1900-04-15T00:00:00Z")
    date_start = record.get("date_start", "")
    if date_start and len(date_start) >= 4:
        return date_start[:4]

    # Try date_iso (e.g. ["1900-04-15"])
    date_iso = record.get("date_iso", [])
    if date_iso and isinstance(date_iso, list) and date_iso[0] and len(date_iso[0]) >= 4:
        return date_iso[0][:4]

    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Split bpl_checkpoint.json by year")
    parser.add_argument("--input", default="bpl_checkpoint.json", help="Path to checkpoint file")
    parser.add_argument("--outdir", default="chunks", help="Output directory for split files")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path} ...")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    processed_ids = set(data.get("processed_ids", []))
    print(f"Total records: {len(records)}, processed_ids: {len(processed_ids)}")

    # Group records by year
    by_year = defaultdict(list)
    for rec in records:
        year = extract_year(rec)
        by_year[year].append(rec)

    # Build a set of processed_ids per year (based on record_id in each group)
    print(f"\nFound {len(by_year)} year groups:")
    for year in sorted(by_year.keys()):
        recs = by_year[year]
        year_ids = [r["record_id"] for r in recs if "record_id" in r]
        out_file = outdir / f"bpl_{year}.json"

        chunk = {
            "records": recs,
            "processed_ids": year_ids,
            "count": len(recs),
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)

        print(f"  {year}: {len(recs):>6} records -> {out_file}")

    print(f"\nDone. {len(by_year)} files written to {outdir}/")


if __name__ == "__main__":
    main()
