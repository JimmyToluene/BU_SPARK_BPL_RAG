"""Step 1 — Fetch record IDs for each war-period year from Digital Commonwealth."""

import time
from datetime import datetime, timezone
from pathlib import Path

from .shared import log, get_with_retry, save_json, load_json


def run(cfg: dict, periods: list[str]) -> int:
    log.info("=" * 60)
    log.info("STEP 1: Fetch Record IDs")
    log.info("=" * 60)

    retry_cfg = cfg.get("retry", {})
    perf = cfg.get("performance", {})
    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    ids_dir = data_dir / out["ids_subdir"]
    ids_dir.mkdir(parents=True, exist_ok=True)

    request_delay = perf.get("request_delay", 1.5)
    sample = perf.get("sample_per_year")
    war_periods = cfg["war_periods"]
    newspapers = cfg["newspapers"]

    grand_total = 0

    for period_key in periods:
        period = war_periods[period_key]
        years = period["years"]
        label = period["label"]
        log.info(f"\n  Period: {label} ({years[0]}-{years[-1]})")

        for year in years:
            for newspaper in newspapers:
                for collection in newspaper["collections"]:
                    ids_file = ids_dir / f"{period_key}_{year}_{collection['name'].replace(' ', '_')}.json"

                    if ids_file.exists():
                        existing = load_json(str(ids_file))
                        count = existing.get("total", 0)
                        log.info(f"    {year} -- already have {count} IDs, skipping")
                        grand_total += count
                        continue

                    base_url = collection["url"]
                    url = (
                        f"{base_url}"
                        f"&range%5Bdate_facet_yearly_itim%5D%5Bbegin%5D={year}"
                        f"&range%5Bdate_facet_yearly_itim%5D%5Bend%5D={year}"
                    )

                    ids = []
                    page = 1
                    while True:
                        page_url = f"{url}&per_page=100&page={page}"
                        resp = get_with_retry(page_url, retry_cfg)
                        if resp is None:
                            log.error(f"    {year} -- failed at page {page}")
                            break

                        data = resp.json()
                        docs = data.get("data", [])
                        meta = data.get("meta", {}).get("pages", {})
                        total = meta.get("total_count", "?")
                        last_page = meta.get("last_page?", False)

                        if page == 1:
                            log.info(f"    {year} -- {total} records in API")
                            if not docs:
                                log.warning(f"    {year} -- no records found")
                                break

                        for doc in docs:
                            ids.append(doc.get("id", ""))

                        if last_page:
                            break
                        page += 1
                        time.sleep(request_delay)

                    # Deduplicate
                    seen = set()
                    unique = []
                    for rid in ids:
                        if rid not in seen:
                            seen.add(rid)
                            unique.append({
                                "id": rid,
                                "newspaper": newspaper["title"],
                                "collection": collection["name"],
                            })

                    if sample and len(unique) > sample:
                        log.info(f"    {year} -- sampling {sample} of {len(unique)} records")
                        unique = unique[:sample]

                    output = {
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "period": period_key,
                        "year": year,
                        "newspaper": newspaper["title"],
                        "collection": collection["name"],
                        "total": len(unique),
                        "records": unique,
                    }
                    save_json(str(ids_file), output)
                    log.info(f"    {year} -- saved {len(unique)} IDs")
                    grand_total += len(unique)

    log.info(f"\n  Step 1 complete: {grand_total} total IDs")
    return grand_total
