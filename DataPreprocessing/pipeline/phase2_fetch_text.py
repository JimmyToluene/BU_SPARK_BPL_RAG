"""
Phase 2: Fetch Full Text + Metadata
─────────────────────────────────────
Reads bpl_ids.json (from Phase 1) and for each record:
  1. Fetches full metadata from individual record API
  2. Fetches OCR full text
  3. Cleans text
  4. Saves to per-year JSON files

Input  : config.yaml, bpl_ids.json
Output : bpl_checkpoint.json       (rolling, stores processed IDs only)
         bpl_clean_dataset/<year>.json  (one file per year)

Usage:
  python phase2_fetch_text.py
  python phase2_fetch_text.py --config config.yaml
"""

import requests
import re
import json
import time
import logging
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

# ── JSON helpers ─────────────────────────────────────────────────────────────

def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path: str):
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_file: str):
    data = load_json(checkpoint_file)
    if data:
        processed = set(data.get("processed_ids", []))
        log.info(f"Checkpoint loaded: {len(processed)} IDs already processed, resuming ...")
        return processed
    return set()

def save_checkpoint(checkpoint_file: str, processed_ids) -> None:
    save_json(checkpoint_file, {
        "processed_ids": list(processed_ids),
        "saved_at":      datetime.now(timezone.utc).isoformat(),
        "count":         len(processed_ids),
    })
    log.info(f"  Checkpoint saved: {len(processed_ids)} processed IDs")

# ── Per-year file helpers ────────────────────────────────────────────────────

def flush_year_buffers(year_buffers, output_dir: Path) -> int:
    """Append buffered records to per-year JSON files and return count flushed."""
    flushed = 0
    for year, new_records in year_buffers.items():
        if not new_records:
            continue
        year_file = output_dir / f"{year}.json"
        existing = load_json(str(year_file))
        if existing:
            records = existing.get("records", [])
        else:
            records = []
        records.extend(new_records)
        save_json(str(year_file), {
            "year":         year,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total":        len(records),
            "records":      records,
        })
        flushed += len(new_records)
        log.info(f"    {year}.json: +{len(new_records)} records (total {len(records)})")
    return flushed

def extract_year(record: dict) -> int:
    """Extract a single year int from a processed record."""
    year_list = record.get("year", [])
    if year_list:
        return int(year_list[0])
    date_start = record.get("date_start", "")
    if date_start:
        return int(date_start[:4])
    return 0

# ── HTTP helper ───────────────────────────────────────────────────────────────

# Shared session for connection pooling (reuses TCP connections)
_session = requests.Session()

def get_with_retry(url: str, retry_cfg: dict, is_text=False):
    max_retries = retry_cfg.get("max_retries", 5)
    backoff     = retry_cfg.get("backoff", [2, 5, 10, 30, 60])

    for attempt in range(max_retries):
        try:
            resp = _session.get(url, timeout=20)
            if is_text and resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            wait = backoff[min(attempt, len(backoff) - 1)]
            log.warning(f"  Timeout attempt {attempt+1}/{max_retries}, retry in {wait}s ...")
            time.sleep(wait)
        except requests.exceptions.ConnectionError:
            wait = backoff[min(attempt, len(backoff) - 1)]
            log.warning(f"  Connection error attempt {attempt+1}/{max_retries}, retry in {wait}s ...")
            time.sleep(wait)
        except requests.exceptions.HTTPError:
            if resp.status_code >= 500:
                wait = backoff[min(attempt, len(backoff) - 1)]
                log.warning(f"  HTTP {resp.status_code} attempt {attempt+1}/{max_retries}, retry in {wait}s ...")
                time.sleep(wait)
            else:
                return None
        except Exception as e:
            log.error(f"  Unexpected error: {e}")
            return None

    log.error(f"  All retries exhausted: {url}")
    return None

# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", raw)  # fix hyphen line breaks
    text = re.sub(r"\s+", " ", text)                    # collapse whitespace
    text = re.sub(r"[^\w\s.,!?;:'\"\-\u2013\u2014()]", "", text)  # remove junk
    return text.strip()

# Replace before stripping — preserve meaning
REPLACEMENTS = {
    '|':  'I',
    '¦':  'I',
    '©':  'o',
    '°':  'o',
    'ſ':  's',
    '∫':  's',
    '\x00': '',
}

def clean_text_alt(raw: str) -> str:
    text = raw

    # 1. Known character replacements first
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)

    # 2. Rejoin hyphenated line breaks
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # 3. Preserve paragraph breaks, collapse other whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)       # max 2 newlines
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces/tabs only
    text = re.sub(r" *\n *", "\n", text)          # clean spaces around newlines

    # 4. Fix spacing around punctuation
    text = re.sub(r"\s([.,;:!?])", r"\1", text)  # remove space before punctuation
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # 5. Now strip remaining junk (after replacements)
    text = re.sub(r"[^\w\s.,!?;:'\"\-\u2013\u2014()]", "", text)

    return text.strip()

# ── Process one record ────────────────────────────────────────────────────────

BASE_URL = "https://www.digitalcommonwealth.org"
ARK_BASE = "https://ark.digitalcommonwealth.org/ark:/50959"

def process_one(entry: dict, retry_cfg: dict):
    """
    For one record entry {id, newspaper, collection}:
      1. Fetch full metadata -> /search/commonwealth:id.json
      2. Fetch OCR text      -> /ark:/50959/id/text
      3. Clean + return structured record
    """
    record_id  = entry["id"]
    newspaper  = entry["newspaper"]
    collection = entry["collection"]
    short_id   = record_id.replace("commonwealth:", "")

    # 1. Full metadata
    meta_resp = get_with_retry(
        f"{BASE_URL}/search/{record_id}.json",
        retry_cfg
    )
    if meta_resp is None:
        return None

    attrs = meta_resp.json().get("data", {}).get("attributes", {})

    # Check transcription flag
    if not attrs.get("has_transcription_bsi", False):
        return None

    # Small delay between the two API calls to reduce 503s
    time.sleep(0.5)

    # 2. OCR text
    text_resp = get_with_retry(
        f"{ARK_BASE}/{short_id}/text",
        retry_cfg,
        is_text=True
    )
    if text_resp is None:
        return None

    # cleaned = clean_text(text_resp.text)
    raw_text = text_resp.text
    # cleaned_alt = clean_text_alt(text_resp.text)

    # cleaned = text_resp.text

    if len(raw_text) < 50:
        return None

    # 3. Build structured record with correct field names
    return {
        # Identity
        "ark_id":        short_id,
        "record_id":     record_id,
        "source_url":    attrs.get("identifier_uri_ss", ""),
        "iiif_manifest": attrs.get("identifier_iiif_manifest_ss", ""),

        # Newspaper info (from config)
        "newspaper":     newspaper,
        "collection":    collection,

        # Bibliographic (correct field names confirmed from API)
        "title":         attrs.get("title_info_primary_tsi", ""),
        "issue_date":    attrs.get("title_info_partnum_tsi", ""),    # "April 15, 1900"
        "date_iso":      attrs.get("date_edtf_ssm", [""]),           # ["1900-04-15"]
        "date_start":    attrs.get("date_start_dtsi", ""),           # "1900-04-15T00:00:00Z"
        "year":          attrs.get("date_facet_yearly_itim", []),    # [1900]
        "publisher":     attrs.get("publisher_tsim", []),
        "place":         attrs.get("publication_place_tsim", []),
        "language":      attrs.get("language_ssim", []),
        "institution":   attrs.get("institution_name_ssi", ""),

        # Pages
        "page_count":    len(attrs.get("filenames_ssim", [])),
        "pages":         attrs.get("filenames_ssim", []),

        # Subject
        "topics":        attrs.get("subject_topic_tsim", []),
        "geography":     attrs.get("subject_geographic_ssim", []),

        # Content
        # "clean_text":    cleaned,
        # "clean_text_alt": cleaned_alt,
        "raw_text": raw_text,
        "char_count":    len(raw_text),

        # Tracking
        "ingested_at":   datetime.now(timezone.utc).isoformat(),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    cfg = load_config(config_path)

    retry_cfg       = cfg.get("retry", {})
    perf            = cfg.get("performance", {})
    output_cfg      = cfg.get("output", {})

    max_workers      = perf.get("max_workers", 4)
    checkpoint_every = perf.get("checkpoint_every", 100)
    ids_filename     = output_cfg.get("ids_file", "bpl_ids.json")
    checkpoint_name  = output_cfg.get("checkpoint_file", "bpl_checkpoint.json")
    dataset_file     = output_cfg.get("dataset_file", "bpl_clean_dataset.json")

    # Root output directory
    output_dir = Path(Path(dataset_file).stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 2: Fetch Full Text + Metadata")
    log.info(f"  Output root : {output_dir}/")
    log.info(f"  Workers     : {max_workers}")
    log.info("=" * 60)

    # Discover collection folders created by Phase 1
    collection_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / ids_filename).exists()
    ])

    if not collection_dirs:
        log.error(f"No collection folders with '{ids_filename}' found in {output_dir}/. Run Phase 1 first.")
        return

    # Process each collection folder
    for coll_dir in collection_dirs:
        ids_file        = coll_dir / ids_filename
        checkpoint_dir  = coll_dir / "temp"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / checkpoint_name

        log.info("-" * 60)
        log.info(f"Collection : {coll_dir.name}")
        log.info(f"  IDs file   : {ids_file}")
        log.info(f"  Checkpoint : {checkpoint_file}")

        # Load IDs from Phase 1
        ids_data = load_json(str(ids_file))
        if not ids_data:
            log.error(f"  Cannot read '{ids_file}'. Skipping.")
            continue

        all_entries = ids_data.get("records", [])
        log.info(f"  Total records : {len(all_entries)}")

        # Load checkpoint (resume support)
        processed_ids = load_checkpoint(str(checkpoint_file))

        # Filter to unprocessed only
        todo = [e for e in all_entries if e["id"] not in processed_ids]

        log.info(f"  Already processed : {len(processed_ids)}")
        log.info(f"  Remaining         : {len(todo)}")

        if not todo:
            log.info("  All records already processed!")
            continue

        start        = time.time()
        skipped      = 0
        total_saved  = 0
        pending_save = 0
        year_buffers = defaultdict(list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_one, entry, retry_cfg): entry
                for entry in todo
            }

            for i, future in enumerate(as_completed(futures), start=1):
                entry  = futures[future]
                result = future.result()

                processed_ids.add(entry["id"])
                pending_save += 1

                if result:
                    year = extract_year(result)
                    year_buffers[year].append(result)
                    total_saved += 1
                else:
                    skipped += 1

                # Rolling checkpoint
                if pending_save >= checkpoint_every:
                    log.info(f"  Flushing {sum(len(v) for v in year_buffers.values())} records ...")
                    flush_year_buffers(year_buffers, coll_dir)
                    year_buffers.clear()
                    save_checkpoint(str(checkpoint_file), processed_ids)
                    pending_save = 0

                # Progress log every 10 records
                if i % 10 == 0 or i == len(todo):
                    elapsed  = time.time() - start
                    rate     = i / elapsed if elapsed > 0 else 0
                    eta_mins = (len(todo) - i) / rate / 60 if rate > 0 else 0
                    log.info(
                        f"  [{i}/{len(todo)}] "
                        f"saved={total_saved} skipped={skipped} "
                        f"| {rate:.1f} rec/s | ETA: {eta_mins:.1f} min"
                    )

        # Final flush
        if year_buffers:
            log.info(f"  Final flush: {sum(len(v) for v in year_buffers.values())} records ...")
            flush_year_buffers(year_buffers, coll_dir)
            year_buffers.clear()
        save_checkpoint(str(checkpoint_file), processed_ids)
        elapsed_mins = (time.time() - start) / 60
        log.info(f"  Done in {elapsed_mins:.1f} min.")

    # Summary across all collections
    grand_total = 0
    all_year_files = sorted(output_dir.glob("*/*.json"))
    # Exclude ids files and temp/ files
    all_year_files = [f for f in all_year_files if f.parent.name != "temp" and f.name != ids_filename]

    for yf in all_year_files:
        data = load_json(str(yf))
        if data:
            grand_total += data.get("total", 0)

    log.info("=" * 60)
    log.info(f"Phase 2 complete.")
    log.info(f"  Total records saved : {grand_total}")
    for yf in all_year_files:
        data = load_json(str(yf))
        if data:
            log.info(f"    {yf.parent.name}/{yf.name}: {data.get('total', 0)} records")
    log.info("=" * 60)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Fetch BPL text + metadata")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run(args.config)
