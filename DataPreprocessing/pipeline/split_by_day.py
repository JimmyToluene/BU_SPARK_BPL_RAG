"""
Split Year Data into Single-Day Files
──────────────────────────────────────
Reads per-year JSON files from Phase 2 output and splits each into
one JSON file per day.

Input  : bpl_clean_dataset/{Title}_{Collection}/{year}.json
Output : bpl_clean_dataset/{Title}_{Collection}/{year}_by_single_day/{date}.json

Usage:
  python -m pipeline.split_by_day
  python -m pipeline.split_by_day --config config.yaml
"""

import json
import logging
import argparse
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str):
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None

def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_date(record: dict) -> str:
    """Extract YYYY-MM-DD date string from a record."""
    date_iso = record.get("date_iso", [])
    if date_iso and date_iso[0]:
        return date_iso[0][:10]
    date_start = record.get("date_start", "")
    if date_start:
        return date_start[:10]
    return "unknown"

# ── Main ──────────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_cfg   = cfg.get("output", {})
    dataset_file = output_cfg.get("dataset_file", "bpl_clean_dataset.json")
    ids_filename = output_cfg.get("ids_file", "bpl_ids.json")
    output_dir   = Path(Path(dataset_file).stem)

    if not output_dir.exists():
        log.error(f"Output directory '{output_dir}' not found. Run Phase 1 & 2 first.")
        return

    # Find collection folders
    collection_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / ids_filename).exists()
    ])

    if not collection_dirs:
        log.error(f"No collection folders found in {output_dir}/.")
        return

    log.info("=" * 60)
    log.info("Split Year Data → Single-Day Files")
    log.info("=" * 60)

    for coll_dir in collection_dirs:
        # Find year JSON files (match {year}.json, skip bpl_ids.json etc.)
        year_files = sorted([
            f for f in coll_dir.glob("*.json")
            if re.match(r"^\d{4}\.json$", f.name)
        ])

        if not year_files:
            log.info(f"  {coll_dir.name}: no year files found, skipping.")
            continue

        log.info(f"\n  Collection: {coll_dir.name}")

        for year_file in year_files:
            year_data = load_json(str(year_file))
            if not year_data:
                continue

            year = year_data.get("year", year_file.stem)
            records = year_data.get("records", [])

            if not records:
                log.info(f"    {year_file.name}: 0 records, skipping.")
                continue

            # Group by date
            by_date = defaultdict(list)
            for record in records:
                date = extract_date(record)
                by_date[date].append(record)

            # Write to {year}_by_single_day/
            day_dir = coll_dir / f"{year}_by_single_day"
            day_dir.mkdir(parents=True, exist_ok=True)

            for date in sorted(by_date.keys()):
                day_records = by_date[date]
                save_json(str(day_dir / f"{date}.json"), {
                    "date":         date,
                    "year":         year,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "total":        len(day_records),
                    "records":      day_records,
                })

            log.info(f"    {year_file.name}: {len(records)} records → "
                     f"{len(by_date)} day files in {year}_by_single_day/")

    log.info("\n" + "=" * 60)
    log.info("Split complete.")
    log.info("=" * 60)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split year data into single-day files")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run(args.config)
