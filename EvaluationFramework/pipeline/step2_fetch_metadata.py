"""Step 2 — Fetch metadata + existing API OCR text for each record."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .shared import (
    log, get_with_retry, load_json, append_jsonl,
    load_checkpoint, save_checkpoint,
)


def _fetch_one(entry: dict, cfg: dict) -> dict | None:
    api = cfg["api"]
    retry_cfg = cfg.get("retry", {})
    record_id = entry["id"]
    short_id = record_id.replace("commonwealth:", "")

    meta_resp = get_with_retry(
        f"{api['base_url']}/search/{record_id}.json", retry_cfg
    )
    if meta_resp is None:
        return None

    attrs = meta_resp.json().get("data", {}).get("attributes", {})
    if not attrs.get("has_transcription_bsi", False):
        return None

    time.sleep(0.5)

    text_resp = get_with_retry(
        f"{api['ark_base']}/{short_id}/text", retry_cfg, is_text=True
    )
    if text_resp is None:
        return None

    raw_text = text_resp.text
    if len(raw_text) < 50:
        return None

    return {
        "ark_id": short_id,
        "record_id": record_id,
        "source_url": attrs.get("identifier_uri_ss", ""),
        "iiif_manifest": attrs.get("identifier_iiif_manifest_ss", ""),
        "newspaper": entry["newspaper"],
        "collection": entry["collection"],
        "title": attrs.get("title_info_primary_tsi", ""),
        "issue_date": attrs.get("title_info_partnum_tsi", ""),
        "date_iso": attrs.get("date_edtf_ssm", [""]),
        "date_start": attrs.get("date_start_dtsi", ""),
        "year": attrs.get("date_facet_yearly_itim", []),
        "publisher": attrs.get("publisher_tsim", []),
        "place": attrs.get("publication_place_tsim", []),
        "institution": attrs.get("institution_name_ssi", ""),
        "page_count": len(attrs.get("filenames_ssim", [])),
        "pages": attrs.get("filenames_ssim", []),
        "topics": attrs.get("subject_topic_tsim", []),
        "geography": attrs.get("subject_geographic_ssim", []),
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def run(cfg: dict, periods: list[str]) -> int:
    log.info("=" * 60)
    log.info("STEP 2: Fetch Metadata + OCR Text")
    log.info("=" * 60)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    ids_dir = data_dir / out["ids_subdir"]
    records_dir = data_dir / out["records_subdir"]
    records_dir.mkdir(parents=True, exist_ok=True)

    perf = cfg.get("performance", {})
    max_workers = perf.get("max_workers", 4)
    checkpoint_every = perf.get("checkpoint_every", 50)
    ckpt_path = str(data_dir / out["checkpoint"])
    ckpt = load_checkpoint(ckpt_path)
    processed_ids = set(ckpt.get("processed_ids", []))

    id_files = sorted(ids_dir.glob("*.json"))
    id_files = [f for f in id_files if any(f.name.startswith(p) for p in periods)]

    total_saved = 0

    for id_file in id_files:
        id_data = load_json(str(id_file))
        if not id_data:
            continue

        year = id_data["year"]
        period = id_data["period"]
        all_entries = id_data.get("records", [])
        todo = [e for e in all_entries if e["id"] not in processed_ids]
        records_file = records_dir / f"{period}_{year}.jsonl"

        log.info(f"  {period} {year}: {len(all_entries)} total, {len(todo)} remaining")
        if not todo:
            continue

        start = time.time()
        batch_saved = 0
        batch_skipped = 0
        pending = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_one, entry, cfg): entry
                for entry in todo
            }
            for i, future in enumerate(as_completed(futures), 1):
                entry = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    log.error(f"    Error {entry['id']}: {e}")
                    result = None

                processed_ids.add(entry["id"])
                pending += 1

                if result:
                    append_jsonl(str(records_file), result)
                    batch_saved += 1
                else:
                    batch_skipped += 1

                if pending >= checkpoint_every:
                    ckpt["processed_ids"] = list(processed_ids)
                    save_checkpoint(ckpt_path, ckpt)
                    pending = 0

                if i % 20 == 0 or i == len(todo):
                    elapsed = time.time() - start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - i) / rate / 60 if rate > 0 else 0
                    log.info(
                        f"    [{i}/{len(todo)}] saved={batch_saved} "
                        f"skip={batch_skipped} | {rate:.1f} rec/s | ETA {eta:.1f}m"
                    )

        ckpt["processed_ids"] = list(processed_ids)
        save_checkpoint(ckpt_path, ckpt)
        total_saved += batch_saved

    log.info(f"\n  Step 2 complete: {total_saved} new records fetched")
    return total_saved
