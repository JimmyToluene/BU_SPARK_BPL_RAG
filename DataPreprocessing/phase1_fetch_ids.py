"""
Phase 1: Fetch Record IDs
─────────────────────────
Reads config.yaml and fetches all record IDs from BPL collections
within the configured date range.

Input  : config.yaml
Output : bpl_ids.json  (list of commonwealth record IDs)

Usage:
  python phase1_fetch_ids.py
  python phase1_fetch_ids.py --config config.yaml
"""

import requests
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Load config ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

# ── HTTP helper ───────────────────────────────────────────────────────────────

def get_with_retry(url: str, retry_cfg: dict):
    max_retries = retry_cfg.get("max_retries", 5)
    backoff     = retry_cfg.get("backoff", [2, 5, 10, 30, 60])

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=20)
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
                log.error(f"  HTTP {resp.status_code} - not retrying")
                return None
        except Exception as e:
            log.error(f"  Unexpected error: {e}")
            return None

    log.error(f"  All {max_retries} retries exhausted: {url}")
    return None

# ── Date filter ───────────────────────────────────────────────────────────────

def build_date_filter(base_url: str, date_start: str, date_end: str) -> str:
    """
    Append date range filter to a collection URL.
    Uses Solr range query format supported by Digital Commonwealth.
    e.g. &range[date_start_dtsi][begin]=1900-01-01T00:00:00Z
         &range[date_start_dtsi][end]=1946-12-31T23:59:59Z
    """
    start_iso = f"{date_start}T00:00:00Z"
    end_iso   = f"{date_end}T23:59:59Z"
    return (
        f"{base_url}"
        f"&range%5Bdate_start_dtsi%5D%5Bbegin%5D={start_iso}"
        f"&range%5Bdate_start_dtsi%5D%5Bend%5D={end_iso}"
    )

# ── Fetch one collection ──────────────────────────────────────────────────────

def fetch_collection_ids(
    name: str,
    base_url: str,
    date_start: str,
    date_end: str,
    retry_cfg: dict,
    request_delay: float,
    max_pages: int | None,
) -> list[str]:
    """
    Paginate through a collection and return all record IDs
    within the date range.
    """
    ids      = []
    page     = 1
    url_with_dates = build_date_filter(base_url, date_start, date_end)

    log.info(f"  Collection: '{name}'")
    log.info(f"  Date range: {date_start} → {date_end}")

    while True:
        if max_pages and page > max_pages:
            log.info(f"    MAX_PAGES={max_pages} reached.")
            break

        url  = f"{url_with_dates}&per_page=100&page={page}"
        resp = get_with_retry(url, retry_cfg)

        if resp is None:
            log.error(f"    Giving up on '{name}' at page {page}.")
            break

        data      = resp.json()
        docs      = data.get("data", [])
        meta      = data.get("meta", {}).get("pages", {})
        total     = meta.get("total_count", "?")
        last_page = meta.get("last_page?", False)

        if page == 1:
            log.info(f"    Records in range: {total}")

        if not docs:
            break

        for doc in docs:
            ids.append(doc.get("id", ""))

        log.info(f"    Page {page}: +{len(docs)} (so far: {len(ids)})")

        if last_page:
            break

        page += 1
        time.sleep(request_delay)

    return ids

# ── Main ──────────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    cfg = load_config(config_path)

    scope       = cfg["scope"]
    date_start  = scope["date_start"]
    date_end    = scope["date_end"]
    retry_cfg   = cfg.get("retry", {})
    perf        = cfg.get("performance", {})
    output_cfg  = cfg.get("output", {})

    request_delay = perf.get("request_delay", 0.3)
    max_pages     = perf.get("max_pages", None)
    ids_file      = output_cfg.get("ids_file", "bpl_ids.json")

    log.info("=" * 60)
    log.info("Phase 1: Fetch Record IDs")
    log.info(f"  Date range : {date_start} → {date_end}")
    log.info(f"  Output     : {ids_file}")
    log.info("=" * 60)

    # Check if already done
    if Path(ids_file).exists():
        existing = json.loads(Path(ids_file).read_text())
        log.info(f"'{ids_file}' already exists with {len(existing)} IDs.")
        log.info("Delete it to re-run Phase 1. Exiting.")
        return

    seen    = set()
    all_ids = []

    for newspaper in cfg["newspapers"]:
        paper_title = newspaper["title"]
        log.info(f"\nNewspaper: {paper_title}")

        for collection in newspaper["collections"]:
            ids = fetch_collection_ids(
                name          = collection["name"],
                base_url      = collection["url"],
                date_start    = date_start,
                date_end      = date_end,
                retry_cfg     = retry_cfg,
                request_delay = request_delay,
                max_pages     = max_pages,
            )

            added = 0
            for rid in ids:
                if rid not in seen:
                    seen.add(rid)
                    all_ids.append({
                        "id":         rid,
                        "newspaper":  paper_title,
                        "collection": collection["name"],
                    })
                    added += 1

            log.info(f"    → {added} unique IDs added")

    # Save output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_start":   date_start,
        "date_end":     date_end,
        "total":        len(all_ids),
        "records":      all_ids,
    }

    Path(ids_file).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    log.info("=" * 60)
    log.info(f"Phase 1 complete.")
    log.info(f"  Total unique IDs : {len(all_ids)}")
    log.info(f"  Saved to         : {ids_file}")
    log.info("=" * 60)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Fetch BPL record IDs")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run(args.config)
