#!/usr/bin/env python3
"""
build_war_period_gt.py — War-Period Pseudo Ground Truth Pipeline
================================================================
BU Spark Student Team — BPL RAG Evaluation Framework

End-to-end pipeline to build precise Q&A ground truth pairs from
BPL newspaper data for WWI (1914-1918) and WWII (1939-1945).

Uses dots.ocr (rednote-hilab/dots.ocr) — a 3B-parameter VLM that
replaces the entire Tesseract + Tier 1/2/3 cleaning chain with a
single end-to-end model that natively handles multi-column newspaper
layout, reading order, and degraded historical scans.

Seven steps, each reads from / writes to disk so they can run
independently or as a single pipeline:

  Step 1  Fetch record IDs for each war-period year
  Step 2  Fetch metadata + existing OCR text per record
  Step 3  Download original page images via IIIF manifests
  Step 4  Run dots.ocr VLM on downloaded page images
  Step 5  Merge OCR texts, clean, and extract war-related articles
  Step 6  Generate Q&A pairs (template-based + LLM-powered)
  Step 7  Export final JSONL with ground truths

Dependencies:
  pip install requests pyyaml Pillow openai

  dots.ocr (Step 4) — two modes:
    vLLM server (recommended):
      pip install vllm
      vllm serve rednote-hilab/dots.ocr.1.5-3B \
        --dtype bfloat16 --max-model-len 8192
    Transformers (in-process):
      pip install transformers torch accelerate

Usage:
  # Full pipeline (all 7 steps)
  python build_war_period_gt.py

  # Run specific step(s)
  python build_war_period_gt.py --steps 1 2
  python build_war_period_gt.py --steps 3 4 5 6 7

  # Quick test (5 records per year, skip IIIF/OCR)
  python build_war_period_gt.py --sample 5 --steps 1 2 5 6 7

  # Custom config
  python build_war_period_gt.py --config my_config.yaml

  # Single war period
  python build_war_period_gt.py --period wwi
  python build_war_period_gt.py --period wwii
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml

# ── Optional imports (graceful degradation) ──────────────────────────────────

try:
    from PIL import Image
    import base64 as _base64
    import io as _io

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("war_gt")

# ── Shared HTTP session ──────────────────────────────────────────────────────

_session = requests.Session()
_session.headers.update({"User-Agent": "BPL-RAG-EvalPipeline/1.0"})

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def append_jsonl(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_with_retry(url: str, retry_cfg: dict, timeout: int = 20, is_text: bool = False):
    max_retries = retry_cfg.get("max_retries", 5)
    backoff = retry_cfg.get("backoff", [2, 5, 10, 30, 60])

    for attempt in range(max_retries):
        try:
            resp = _session.get(url, timeout=timeout)
            if is_text and resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            wait = backoff[min(attempt, len(backoff) - 1)]
            log.warning(f"  Timeout {attempt + 1}/{max_retries}, retry in {wait}s …")
            time.sleep(wait)
        except requests.exceptions.ConnectionError:
            wait = backoff[min(attempt, len(backoff) - 1)]
            log.warning(f"  ConnErr {attempt + 1}/{max_retries}, retry in {wait}s …")
            time.sleep(wait)
        except requests.exceptions.HTTPError:
            if resp.status_code >= 500:
                wait = backoff[min(attempt, len(backoff) - 1)]
                log.warning(f"  HTTP {resp.status_code} {attempt + 1}/{max_retries}, retry in {wait}s …")
                time.sleep(wait)
            else:
                return None
        except Exception as e:
            log.error(f"  Unexpected error: {e}")
            return None

    log.error(f"  All retries exhausted: {url}")
    return None


# ── Checkpoint ───────────────────────────────────────────────────────────────


def load_checkpoint(path: str) -> dict:
    data = load_json(path)
    return data if data else {"processed_ids": [], "completed_steps": {}}


def save_checkpoint(path: str, ckpt: dict):
    ckpt["saved_at"] = datetime.now(timezone.utc).isoformat()
    save_json(path, ckpt)


# ── Text cleaning (from DataPreprocessing) ───────────────────────────────────

REPLACEMENTS = {
    "|": "I", "¦": "I", "©": "o", "°": "o",
    "ſ": "s", "∫": "s", "\x00": "",
}


def clean_text(raw: str) -> str:
    text = raw
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\s([.,;:!?])", r"\1", text)
    text = re.sub(r"[^\w\s.,!?;:'\"\-\u2013\u2014()]", "", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Fetch Record IDs
# ─────────────────────────────────────────────────────────────────────────────


def step1_fetch_ids(cfg: dict, periods: list[str]):
    """Fetch all record IDs for each war-period year from Digital Commonwealth."""
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

                    # Skip if already fetched
                    if ids_file.exists():
                        existing = load_json(str(ids_file))
                        count = existing.get("total", 0)
                        log.info(f"    {year} — already have {count} IDs, skipping")
                        grand_total += count
                        continue

                    # Build date-filtered search URL
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
                            log.error(f"    {year} — failed at page {page}")
                            break

                        data = resp.json()
                        docs = data.get("data", [])
                        meta = data.get("meta", {}).get("pages", {})
                        total = meta.get("total_count", "?")
                        last_page = meta.get("last_page?", False)

                        if page == 1:
                            log.info(f"    {year} — {total} records in API")
                            if not docs:
                                log.warning(f"    {year} — no records found")
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

                    # Apply sampling if configured
                    if sample and len(unique) > sample:
                        log.info(f"    {year} — sampling {sample} of {len(unique)} records")
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
                    log.info(f"    {year} — saved {len(unique)} IDs → {ids_file.name}")
                    grand_total += len(unique)

    log.info(f"\n  Step 1 complete: {grand_total} total IDs across all years")
    return grand_total


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Fetch Metadata + API OCR Text
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_one_record(entry: dict, cfg: dict) -> dict | None:
    """Fetch metadata + OCR text for a single record."""
    api = cfg["api"]
    retry_cfg = cfg.get("retry", {})
    record_id = entry["id"]
    short_id = record_id.replace("commonwealth:", "")

    # 1. Metadata
    meta_url = f"{api['base_url']}/search/{record_id}.json"
    meta_resp = get_with_retry(meta_url, retry_cfg)
    if meta_resp is None:
        return None

    attrs = meta_resp.json().get("data", {}).get("attributes", {})
    if not attrs.get("has_transcription_bsi", False):
        return None

    time.sleep(0.5)

    # 2. OCR text
    text_url = f"{api['ark_base']}/{short_id}/text"
    text_resp = get_with_retry(text_url, retry_cfg, is_text=True)
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


def step2_fetch_metadata(cfg: dict, periods: list[str]):
    """Fetch metadata + OCR text for all records identified in Step 1."""
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

    # Collect all ID files for requested periods
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
                executor.submit(_fetch_one_record, entry, cfg): entry
                for entry in todo
            }

            for i, future in enumerate(as_completed(futures), 1):
                entry = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    log.error(f"    Error processing {entry['id']}: {e}")
                    result = None

                processed_ids.add(entry["id"])
                pending += 1

                if result:
                    append_jsonl(str(records_file), result)
                    batch_saved += 1
                else:
                    batch_skipped += 1

                # Checkpoint
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

        # Final checkpoint
        ckpt["processed_ids"] = list(processed_ids)
        save_checkpoint(ckpt_path, ckpt)
        total_saved += batch_saved
        log.info(f"  {period} {year}: done — {batch_saved} saved, {batch_skipped} skipped")

    log.info(f"\n  Step 2 complete: {total_saved} new records fetched")
    return total_saved


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Download IIIF Page Images
# ─────────────────────────────────────────────────────────────────────────────


def _parse_iiif_manifest(manifest_url: str, retry_cfg: dict) -> list[str]:
    """Parse a IIIF manifest and return a list of page image base URLs."""
    resp = get_with_retry(manifest_url, retry_cfg, timeout=30)
    if resp is None:
        return []

    try:
        manifest = resp.json()
    except Exception:
        return []

    image_urls = []
    sequences = manifest.get("sequences", [])
    if not sequences:
        return []

    for canvas in sequences[0].get("canvases", []):
        images = canvas.get("images", [])
        if not images:
            continue
        resource = images[0].get("resource", {})

        # Try service endpoint first (allows size control)
        service = resource.get("service", {})
        service_id = service.get("@id", "")
        if service_id:
            image_urls.append(service_id)
        else:
            # Fall back to direct resource URL
            res_id = resource.get("@id", "")
            if res_id:
                image_urls.append(res_id)

    return image_urls


def _download_page_image(
    service_url: str,
    save_path: Path,
    width: int,
    retry_cfg: dict,
    iiif_timeout: int,
):
    """Download a single page image from a IIIF service endpoint."""
    if save_path.exists():
        return True

    # Construct IIIF Image API URL: /full/{width},/0/default.jpg
    if "/full/" in service_url:
        # Already a full image URL — use as-is
        img_url = service_url
    else:
        img_url = f"{service_url}/full/{width},/0/default.jpg"

    resp = get_with_retry(img_url, retry_cfg, timeout=iiif_timeout)
    if resp is None:
        return False

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return True


def step3_download_pages(cfg: dict, periods: list[str]):
    """Download original newspaper page images via IIIF manifests."""
    if not HAS_PIL:
        log.warning("Pillow not installed — step 4 (dots.ocr) needs it. "
                     "Install with: pip install Pillow")

    log.info("=" * 60)
    log.info("STEP 3: Download IIIF Page Images")
    log.info("=" * 60)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    records_dir = data_dir / out["records_subdir"]
    pages_dir = data_dir / out["pages_subdir"]
    pages_dir.mkdir(parents=True, exist_ok=True)

    retry_cfg = cfg.get("retry", {})
    iiif_cfg = cfg.get("iiif", {})
    perf = cfg.get("performance", {})
    width = iiif_cfg.get("image_width", 1500)
    iiif_timeout = iiif_cfg.get("timeout", 30)
    max_pages_per_issue = perf.get("max_pages_per_issue")
    request_delay = perf.get("request_delay", 1.5)

    # Find all record files for requested periods
    record_files = sorted(records_dir.glob("*.jsonl"))
    record_files = [f for f in record_files if any(f.stem.startswith(p) for p in periods)]

    total_pages = 0
    total_issues = 0

    for rec_file in record_files:
        log.info(f"  Processing: {rec_file.name}")
        records = []
        with open(rec_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        for rec in records:
            ark_id = rec["ark_id"]
            manifest_url = rec.get("iiif_manifest", "")
            issue_dir = pages_dir / ark_id

            if not manifest_url:
                log.warning(f"    {ark_id} — no IIIF manifest URL, skipping")
                continue

            # Check if already downloaded (marker file)
            done_marker = issue_dir / ".done"
            if done_marker.exists():
                continue

            # Parse IIIF manifest
            image_urls = _parse_iiif_manifest(manifest_url, retry_cfg)
            if not image_urls:
                log.warning(f"    {ark_id} — could not parse manifest, skipping")
                continue

            # Apply page limit
            if max_pages_per_issue and len(image_urls) > max_pages_per_issue:
                image_urls = image_urls[:max_pages_per_issue]

            issue_dir.mkdir(parents=True, exist_ok=True)
            downloaded = 0

            for idx, img_url in enumerate(image_urls):
                page_path = issue_dir / f"page_{idx + 1:04d}.jpg"
                ok = _download_page_image(img_url, page_path, width, retry_cfg, iiif_timeout)
                if ok:
                    downloaded += 1
                time.sleep(0.3)  # gentle rate limit for image server

            # Write marker
            done_marker.write_text(f"{downloaded} pages\n")
            total_pages += downloaded
            total_issues += 1

            if total_issues % 10 == 0:
                log.info(f"    … {total_issues} issues, {total_pages} pages downloaded")

            time.sleep(request_delay)

    log.info(f"\n  Step 3 complete: {total_issues} issues, {total_pages} pages downloaded")
    return total_pages


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — dots.ocr VLM OCR on Downloaded Pages
# ─────────────────────────────────────────────────────────────────────────────


def _image_to_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return _base64.b64encode(f.read()).decode("utf-8")


def _ocr_page_vllm(image_path: Path, client, model: str, prompt: str,
                    max_tokens: int, timeout: int) -> str:
    """OCR a single page image via vLLM OpenAI-compatible API."""
    img_b64 = _image_to_base64(image_path)
    suffix = image_path.suffix.lstrip(".").lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(suffix, "jpeg")

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{img_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }],
        timeout=timeout,
    )
    return response.choices[0].message.content.strip()


def _ocr_page_transformers(image_path: Path, model, processor, prompt: str,
                           max_tokens: int, device: str) -> str:
    """OCR a single page image via HuggingFace Transformers (in-process)."""
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    # Decode only new tokens (skip the input prompt tokens)
    input_len = inputs["input_ids"].shape[1]
    text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return text.strip()


def _init_dots_ocr(cfg: dict):
    """Initialize dots.ocr backend based on config mode.

    Returns:
        (mode, client_or_model, processor_or_None, device_or_None)
    """
    dots_cfg = cfg.get("dots_ocr", {})
    mode = dots_cfg.get("mode", "vllm")
    model_name = dots_cfg.get("model", "rednote-hilab/dots.ocr.1.5-3B")

    if mode == "vllm":
        if not HAS_OPENAI:
            log.error("openai package required for vLLM mode. Install: pip install openai")
            return None, None, None, None
        base_url = dots_cfg.get("vllm_base_url", "http://localhost:8000/v1")
        client = OpenAI(base_url=base_url, api_key="dummy")
        # Quick health check
        try:
            client.models.list()
            log.info(f"  Connected to vLLM server at {base_url}")
        except Exception as e:
            log.error(f"  Cannot reach vLLM server at {base_url}: {e}")
            log.error(f"  Start it with: vllm serve {model_name} "
                      f"--dtype bfloat16 --max-model-len 8192")
            return None, None, None, None
        return "vllm", client, None, None

    elif mode == "transformers":
        if not HAS_TRANSFORMERS:
            log.error("transformers + torch required. Install: "
                      "pip install transformers torch accelerate")
            return None, None, None, None
        if not HAS_PIL:
            log.error("Pillow required. Install: pip install Pillow")
            return None, None, None, None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            log.warning("  No CUDA GPU detected — dots.ocr will be very slow on CPU")

        log.info(f"  Loading {model_name} on {device} …")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        log.info(f"  Model loaded successfully")
        return "transformers", model, processor, device

    else:
        log.error(f"  Unknown dots_ocr mode: {mode}. Use 'vllm' or 'transformers'.")
        return None, None, None, None


def step4_ocr_pages(cfg: dict, periods: list[str]):
    """Run dots.ocr VLM on downloaded page images for layout-aware OCR."""
    if not HAS_PIL:
        log.error("Pillow is required. Install: pip install Pillow")
        return 0

    log.info("=" * 60)
    log.info("STEP 4: dots.ocr VLM OCR")
    log.info("=" * 60)

    dots_cfg = cfg.get("dots_ocr", {})
    model_name = dots_cfg.get("model", "rednote-hilab/dots.ocr.1.5-3B")
    prompt = dots_cfg.get("prompt", "<ocr>")
    max_tokens = dots_cfg.get("max_tokens", 8192)
    timeout = dots_cfg.get("timeout", 120)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    pages_dir = data_dir / out["pages_subdir"]
    ocr_dir = data_dir / out["ocr_subdir"]
    ocr_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dots.ocr backend
    mode, backend, processor, device = _init_dots_ocr(cfg)
    if mode is None:
        log.error("  dots.ocr initialization failed — cannot proceed")
        return 0

    log.info(f"  Mode    : {mode}")
    log.info(f"  Model   : {model_name}")
    log.info(f"  Prompt  : {prompt}")

    # Find all issue directories
    issue_dirs = sorted([d for d in pages_dir.iterdir() if d.is_dir()])
    total_ocr = 0

    for issue_dir in issue_dirs:
        ark_id = issue_dir.name
        ocr_file = ocr_dir / f"{ark_id}.txt"

        # Skip if already OCR'd
        if ocr_file.exists():
            continue

        page_images = sorted(issue_dir.glob("page_*.jpg"))
        if not page_images:
            continue

        full_text_parts = []
        for page_img in page_images:
            try:
                if mode == "vllm":
                    text = _ocr_page_vllm(
                        page_img, backend, model_name, prompt, max_tokens, timeout
                    )
                else:
                    text = _ocr_page_transformers(
                        page_img, backend, processor, prompt, max_tokens, device
                    )
                full_text_parts.append(text)
            except Exception as e:
                log.warning(f"    OCR failed for {page_img.name}: {e}")
                full_text_parts.append("")

        full_text = "\n\n--- PAGE BREAK ---\n\n".join(full_text_parts)
        ocr_file.write_text(full_text, encoding="utf-8")
        total_ocr += 1

        if total_ocr % 5 == 0:
            log.info(f"    … {total_ocr} issues OCR'd with dots.ocr")

    log.info(f"\n  Step 4 complete: {total_ocr} issues processed with dots.ocr")
    return total_ocr


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Merge OCR, Clean Text, Extract War Articles
# ─────────────────────────────────────────────────────────────────────────────


def _score_segment(text: str, keywords: list[str]) -> int:
    """Score a text segment by war-keyword density."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _split_into_segments(text: str, min_len: int = 200) -> list[str]:
    """Split newspaper text into rough article segments.

    Strategy: split on double-newlines (paragraph breaks), then
    merge consecutive short paragraphs into larger segments.
    """
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    segments = []
    current = ""
    for para in paragraphs:
        if current:
            current += "\n\n" + para
        else:
            current = para
        if len(current) >= min_len:
            segments.append(current)
            current = ""
    if current:
        if segments:
            segments[-1] += "\n\n" + current
        else:
            segments.append(current)

    return segments


def step5_extract_articles(cfg: dict, periods: list[str]):
    """Merge OCR texts, clean, and extract war-related article segments."""
    log.info("=" * 60)
    log.info("STEP 5: Merge OCR + Extract War Articles")
    log.info("=" * 60)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    records_dir = data_dir / out["records_subdir"]
    ocr_dir = data_dir / out["ocr_subdir"]
    articles_dir = data_dir / out["articles_subdir"]
    articles_dir.mkdir(parents=True, exist_ok=True)

    war_keywords = cfg.get("war_keywords", {})

    # Find all record files for requested periods
    record_files = sorted(records_dir.glob("*.jsonl"))
    record_files = [f for f in record_files if any(f.stem.startswith(p) for p in periods)]

    total_articles = 0
    total_records = 0

    for rec_file in record_files:
        period = rec_file.stem.split("_")[0]  # "wwi" or "wwii"
        keywords = war_keywords.get(period, war_keywords.get("wwi", []))

        records = []
        with open(rec_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        log.info(f"  Processing {rec_file.name}: {len(records)} records")

        for rec in records:
            ark_id = rec["ark_id"]

            # Use dots.ocr output if available, otherwise fall back to API OCR
            dots_ocr_path = ocr_dir / f"{ark_id}.txt"
            if dots_ocr_path.exists():
                text = dots_ocr_path.read_text(encoding="utf-8")
                ocr_source = "dots.ocr"
            else:
                text = rec.get("raw_text", "")
                ocr_source = "api"

            if not text or len(text) < 100:
                continue

            # Clean text
            cleaned = clean_text(text)

            # Split into segments and score
            segments = _split_into_segments(cleaned, min_len=200)
            scored = []
            for seg in segments:
                score = _score_segment(seg, keywords)
                if score >= 2:  # at least 2 keyword hits
                    scored.append({"text": seg, "score": score})

            scored.sort(key=lambda x: x["score"], reverse=True)

            if not scored:
                total_records += 1
                continue

            # Keep top segments (max 5 per issue)
            top_segments = scored[:5]

            article_record = {
                "ark_id": ark_id,
                "record_id": rec["record_id"],
                "newspaper": rec["newspaper"],
                "issue_date": rec.get("issue_date", ""),
                "date_iso": rec.get("date_iso", []),
                "year": rec.get("year", []),
                "title": rec.get("title", ""),
                "period": period,
                "ocr_source": ocr_source,
                "total_segments": len(segments),
                "war_segments": len(scored),
                "top_segments": top_segments,
                "full_text_chars": len(cleaned),
            }

            articles_file = articles_dir / f"{period}_{rec.get('year', [0])[0] if rec.get('year') else 'unknown'}.jsonl"
            append_jsonl(str(articles_file), article_record)
            total_articles += 1
            total_records += 1

        log.info(f"    → {total_articles} issues with war content so far")

    log.info(f"\n  Step 5 complete: {total_articles} issues with war articles "
             f"out of {total_records} total")
    return total_articles


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Generate Q&A Pairs
# ─────────────────────────────────────────────────────────────────────────────

# Template-based questions (used when LLM is not available)
QA_TEMPLATES = [
    {
        "template": "What did the {newspaper} report about the war on {date}?",
        "question_type": "explanatory",
    },
    {
        "template": "Find newspaper coverage of war events from {date} in the {newspaper}.",
        "question_type": "document",
    },
    {
        "template": "What war-related news appeared in the {newspaper} on {date}?",
        "question_type": "explanatory",
    },
    {
        "template": "Show me {newspaper} articles from {date} about military operations.",
        "question_type": "document",
    },
    {
        "template": "How did the {newspaper} cover the war effort on {date}?",
        "question_type": "explanatory",
    },
]

LLM_SYSTEM_PROMPT = """You are a historical newspaper analyst for the Boston Public Library.
You generate evaluation Q&A pairs from actual newspaper content.

Given a newspaper excerpt from a specific date, create ONE focused question that
the excerpt directly answers, plus a concise 2-4 sentence answer.

Rules:
- The question must be specific enough that the given newspaper excerpt is a clear ground truth
- The answer must be based ONLY on the excerpt content
- Include specific dates, names, and events mentioned in the text
- Return JSON only — no preamble, no markdown fences

Response format:
{
  "question": "Your specific question about the war-era content",
  "question_type": "explanatory or document",
  "answer": "Your 2-4 sentence answer citing the newspaper text",
  "confidence": "high|medium|low"
}"""


def _generate_qa_template(article: dict, template_idx: int) -> dict:
    """Generate a Q&A pair using a template (no LLM needed)."""
    tmpl = QA_TEMPLATES[template_idx % len(QA_TEMPLATES)]
    newspaper = article["newspaper"]
    date = article.get("issue_date", "unknown date")
    top_seg = article["top_segments"][0]

    question = tmpl["template"].format(newspaper=newspaper, date=date)

    # Build answer from the top-scoring segment
    answer_text = top_seg["text"][:500]
    answer = (
        f"Based on the {newspaper} from {date}: {answer_text}..."
    )

    return {
        "question": question,
        "question_type": tmpl["question_type"],
        "ground_truths": [
            {
                "title": f"{article['title']}, {date}",
                "ark_id": f"commonwealth:{article['ark_id']}",
            }
        ],
        "answer": answer,
        "confidence": "medium",
        "notes": f"pseudoGT_template | war_score={top_seg['score']} | ocr={article['ocr_source']}",
        "period": article["period"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _generate_qa_llm(article: dict, client, model: str) -> dict | None:
    """Generate a Q&A pair using an LLM for higher quality."""
    top_seg = article["top_segments"][0]
    newspaper = article["newspaper"]
    date = article.get("issue_date", "unknown date")

    user_prompt = (
        f"Newspaper: {newspaper}\n"
        f"Date: {date}\n"
        f"ark_id: commonwealth:{article['ark_id']}\n\n"
        f"Excerpt:\n{top_seg['text'][:2000]}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=800,
            temperature=0.3,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)
    except Exception as e:
        log.warning(f"    LLM failed for {article['ark_id']}: {e}")
        return None

    return {
        "question": result.get("question", ""),
        "question_type": result.get("question_type", "explanatory"),
        "ground_truths": [
            {
                "title": f"{article['title']}, {date}",
                "ark_id": f"commonwealth:{article['ark_id']}",
            }
        ],
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", "medium"),
        "notes": f"pseudoGT_llm | war_score={top_seg['score']} | ocr={article['ocr_source']}",
        "period": article["period"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def step6_generate_qa(cfg: dict, periods: list[str]):
    """Generate Q&A pairs from extracted war articles."""
    log.info("=" * 60)
    log.info("STEP 6: Generate Q&A Pairs")
    log.info("=" * 60)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    articles_dir = data_dir / out["articles_subdir"]
    qa_path = data_dir / out["qa_output"]

    llm_cfg = cfg.get("llm", {})
    model = llm_cfg.get("model", "gpt-4o")

    # Check for LLM availability
    use_llm = False
    client = None
    if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            use_llm = True
            log.info(f"  Using LLM ({model}) for Q&A generation")
        except Exception:
            pass

    if not use_llm:
        log.info("  Using template-based Q&A generation (no LLM)")
        log.info("  Set OPENAI_API_KEY and install openai for LLM-powered generation")

    # Load existing Q&As to avoid duplicates
    existing_arks = set()
    if qa_path.exists():
        with open(qa_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    for gt in obj.get("ground_truths", []):
                        existing_arks.add(gt.get("ark_id", ""))

    # Find article files for requested periods
    article_files = sorted(articles_dir.glob("*.jsonl"))
    article_files = [f for f in article_files if any(f.stem.startswith(p) for p in periods)]

    total_qa = 0
    template_idx = 0

    for art_file in article_files:
        articles = []
        with open(art_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    articles.append(json.loads(line))

        log.info(f"  Processing {art_file.name}: {len(articles)} articles")

        for article in articles:
            ark_full = f"commonwealth:{article['ark_id']}"
            if ark_full in existing_arks:
                continue

            if not article.get("top_segments"):
                continue

            # Generate Q&A
            qa = None
            if use_llm:
                qa = _generate_qa_llm(article, client, model)

            if qa is None:
                qa = _generate_qa_template(article, template_idx)
                template_idx += 1

            if qa and qa.get("question"):
                append_jsonl(str(qa_path), qa)
                existing_arks.add(ark_full)
                total_qa += 1

                if total_qa % 10 == 0:
                    log.info(f"    … {total_qa} Q&A pairs generated")

    log.info(f"\n  Step 6 complete: {total_qa} new Q&A pairs generated → {qa_path}")
    return total_qa


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Export Final JSONL (merge + deduplicate)
# ─────────────────────────────────────────────────────────────────────────────


def step7_export(cfg: dict, periods: list[str]):
    """Export final deduplicated Q&A dataset, with summary statistics."""
    log.info("=" * 60)
    log.info("STEP 7: Export Final JSONL + Summary")
    log.info("=" * 60)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    qa_path = data_dir / out["qa_output"]

    if not qa_path.exists():
        log.error(f"  No Q&A file found at {qa_path}")
        return 0

    # Load all Q&A pairs
    qa_pairs = []
    seen_questions = set()
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                q = obj.get("question", "")
                if q not in seen_questions:
                    seen_questions.add(q)
                    qa_pairs.append(obj)

    # Filter to requested periods
    filtered = [
        qa for qa in qa_pairs
        if qa.get("period", "") in periods or not qa.get("period")
    ]

    # Write deduplicated output
    dedup_path = data_dir / f"war_period_qa_final.jsonl"
    with open(dedup_path, "w", encoding="utf-8") as f:
        for qa in filtered:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # Summary
    by_period = defaultdict(int)
    by_type = defaultdict(int)
    by_confidence = defaultdict(int)
    by_source = defaultdict(int)

    for qa in filtered:
        by_period[qa.get("period", "unknown")] += 1
        by_type[qa.get("question_type", "unknown")] += 1
        by_confidence[qa.get("confidence", "unknown")] += 1
        notes = qa.get("notes", "")
        if "llm" in notes:
            by_source["llm"] += 1
        else:
            by_source["template"] += 1

    log.info(f"  Total Q&A pairs: {len(filtered)}")
    log.info(f"  Output: {dedup_path}")
    log.info(f"\n  By period:     {dict(by_period)}")
    log.info(f"  By type:       {dict(by_type)}")
    log.info(f"  By confidence: {dict(by_confidence)}")
    log.info(f"  By source:     {dict(by_source)}")

    # Also write a human-readable summary
    summary_path = data_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("War-Period Pseudo Ground Truth — Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Total Q&A pairs: {len(filtered)}\n\n")
        f.write(f"By period:     {dict(by_period)}\n")
        f.write(f"By type:       {dict(by_type)}\n")
        f.write(f"By confidence: {dict(by_confidence)}\n")
        f.write(f"By source:     {dict(by_source)}\n\n")
        f.write("Sample questions:\n")
        for qa in filtered[:5]:
            f.write(f"  Q: {qa['question'][:80]}…\n")
            f.write(f"  A: {qa['answer'][:80]}…\n")
            f.write(f"  GT: {qa['ground_truths']}\n\n")

    log.info(f"  Summary: {summary_path}")
    return len(filtered)


# ─────────────────────────────────────────────────────────────────────────────
#  ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

STEP_FUNCTIONS = {
    1: step1_fetch_ids,
    2: step2_fetch_metadata,
    3: step3_download_pages,
    4: step4_ocr_pages,
    5: step5_extract_articles,
    6: step6_generate_qa,
    7: step7_export,
}

STEP_NAMES = {
    1: "Fetch Record IDs",
    2: "Fetch Metadata + OCR Text",
    3: "Download IIIF Page Images",
    4: "dots.ocr VLM OCR",
    5: "Merge OCR + Extract War Articles",
    6: "Generate Q&A Pairs",
    7: "Export Final JSONL",
}


def run_pipeline(config_path: str, steps: list[int], period: str | None, sample: int | None):
    cfg = load_config(config_path)

    # Override sample if provided via CLI
    if sample is not None:
        cfg.setdefault("performance", {})["sample_per_year"] = sample

    # Determine which war periods to process
    available = list(cfg["war_periods"].keys())
    if period:
        if period not in available:
            log.error(f"Unknown period '{period}'. Available: {available}")
            sys.exit(1)
        periods = [period]
    else:
        periods = available

    # Dependency check
    log.info("=" * 60)
    log.info("  War-Period Pseudo Ground Truth Pipeline")
    log.info("=" * 60)
    log.info(f"  Config    : {config_path}")
    log.info(f"  Periods   : {periods}")
    log.info(f"  Steps     : {steps}")
    log.info(f"  Sample    : {sample or 'full dataset'}")
    log.info(f"  PIL       : {'yes' if HAS_PIL else 'NO — steps 3-4 limited'}")
    dots_mode = cfg.get("dots_ocr", {}).get("mode", "vllm")
    if dots_mode == "vllm":
        dots_ready = HAS_OPENAI
        dots_label = f"vllm ({'ready' if dots_ready else 'NO — need openai package'})"
    else:
        dots_ready = HAS_TRANSFORMERS and HAS_PIL
        dots_label = f"transformers ({'ready' if dots_ready else 'NO — need transformers+torch'})"
    log.info(f"  dots.ocr  : {dots_label}")
    log.info(f"  OpenAI    : {'yes' if HAS_OPENAI and os.environ.get('OPENAI_API_KEY') else 'NO — step 6 uses templates'}")
    log.info("=" * 60)

    for step_num in steps:
        if step_num not in STEP_FUNCTIONS:
            log.error(f"Unknown step {step_num}. Valid: 1-7")
            continue

        log.info(f"\n{'▶':} Running Step {step_num}: {STEP_NAMES[step_num]}")
        start = time.time()
        result = STEP_FUNCTIONS[step_num](cfg, periods)
        elapsed = time.time() - start
        log.info(f"  Step {step_num} finished in {elapsed:.1f}s (result: {result})\n")

    log.info("=" * 60)
    log.info("  Pipeline complete!")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="War-Period Pseudo Ground Truth Pipeline — BPL RAG Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all 7 steps, both war periods)
  python build_war_period_gt.py

  # Quick test: 5 records per year, skip IIIF/OCR
  python build_war_period_gt.py --sample 5 --steps 1 2 5 6 7

  # Only fetch IDs and metadata
  python build_war_period_gt.py --steps 1 2

  # Only WWII
  python build_war_period_gt.py --period wwii

  # Download pages + OCR only (after steps 1-2 are done)
  python build_war_period_gt.py --steps 3 4
        """,
    )
    parser.add_argument(
        "--config", default="war_period_config.yaml",
        help="Path to config YAML (default: war_period_config.yaml)",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7],
        help="Which steps to run (default: all). E.g. --steps 1 2 5 6 7",
    )
    parser.add_argument(
        "--period", choices=["wwi", "wwii"], default=None,
        help="Only process one war period (default: both)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N records per year for quick testing",
    )
    args = parser.parse_args()

    run_pipeline(args.config, sorted(args.steps), args.period, args.sample)


if __name__ == "__main__":
    main()
