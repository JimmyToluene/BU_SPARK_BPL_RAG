"""Step 3 — Download original newspaper page images via IIIF manifests."""

import time
from pathlib import Path

from .shared import log, get_with_retry, read_jsonl


def _parse_iiif_manifest(manifest_url: str, retry_cfg: dict) -> list[str]:
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
        service = resource.get("service", {})
        service_id = service.get("@id", "")
        if service_id:
            image_urls.append(service_id)
        else:
            res_id = resource.get("@id", "")
            if res_id:
                image_urls.append(res_id)

    return image_urls


def _download_page(service_url: str, save_path: Path, width: int,
                   retry_cfg: dict, iiif_timeout: int) -> bool:
    if save_path.exists():
        return True

    if "/full/" in service_url:
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


def run(cfg: dict, periods: list[str]) -> int:
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

    record_files = sorted(records_dir.glob("*.jsonl"))
    record_files = [f for f in record_files if any(f.stem.startswith(p) for p in periods)]

    total_pages = 0
    total_issues = 0

    for rec_file in record_files:
        log.info(f"  Processing: {rec_file.name}")
        records = read_jsonl(str(rec_file))

        for rec in records:
            ark_id = rec["ark_id"]
            manifest_url = rec.get("iiif_manifest", "")
            issue_dir = pages_dir / ark_id

            if not manifest_url:
                continue

            done_marker = issue_dir / ".done"
            if done_marker.exists():
                continue

            image_urls = _parse_iiif_manifest(manifest_url, retry_cfg)
            if not image_urls:
                log.warning(f"    {ark_id} -- no images in manifest")
                continue

            if max_pages_per_issue and len(image_urls) > max_pages_per_issue:
                image_urls = image_urls[:max_pages_per_issue]

            issue_dir.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            for idx, img_url in enumerate(image_urls):
                page_path = issue_dir / f"page_{idx + 1:04d}.jpg"
                if _download_page(img_url, page_path, width, retry_cfg, iiif_timeout):
                    downloaded += 1
                time.sleep(0.3)

            done_marker.write_text(f"{downloaded} pages\n")
            total_pages += downloaded
            total_issues += 1

            if total_issues % 10 == 0:
                log.info(f"    ... {total_issues} issues, {total_pages} pages")

            time.sleep(request_delay)

    log.info(f"\n  Step 3 complete: {total_issues} issues, {total_pages} pages")
    return total_pages
