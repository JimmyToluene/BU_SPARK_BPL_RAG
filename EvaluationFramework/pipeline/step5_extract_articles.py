"""Step 5 — Merge OCR texts, clean, and extract war-related article segments."""

import re
from pathlib import Path

from .shared import log, read_jsonl, append_jsonl, clean_text


def _score_segment(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _split_into_segments(text: str, min_len: int = 200) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    segments = []
    current = ""
    for para in paragraphs:
        current = f"{current}\n\n{para}" if current else para
        if len(current) >= min_len:
            segments.append(current)
            current = ""
    if current:
        if segments:
            segments[-1] += "\n\n" + current
        else:
            segments.append(current)
    return segments


def run(cfg: dict, periods: list[str]) -> int:
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

    record_files = sorted(records_dir.glob("*.jsonl"))
    record_files = [f for f in record_files if any(f.stem.startswith(p) for p in periods)]

    total_articles = 0
    total_records = 0

    for rec_file in record_files:
        period = rec_file.stem.split("_")[0]
        keywords = war_keywords.get(period, war_keywords.get("wwi", []))
        records = read_jsonl(str(rec_file))
        log.info(f"  Processing {rec_file.name}: {len(records)} records")

        for rec in records:
            ark_id = rec["ark_id"]

            # Prefer dots.ocr output, fall back to API OCR
            dots_path = ocr_dir / f"{ark_id}.txt"
            if dots_path.exists():
                text = dots_path.read_text(encoding="utf-8")
                ocr_source = "dots.ocr"
            else:
                text = rec.get("raw_text", "")
                ocr_source = "api"

            if not text or len(text) < 100:
                continue

            cleaned = clean_text(text)
            segments = _split_into_segments(cleaned)
            scored = [
                {"text": seg, "score": sc}
                for seg in segments
                if (sc := _score_segment(seg, keywords)) >= 2
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)

            if not scored:
                total_records += 1
                continue

            article = {
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
                "top_segments": scored[:5],
                "full_text_chars": len(cleaned),
            }

            year_val = rec.get("year", [0])[0] if rec.get("year") else "unknown"
            articles_file = articles_dir / f"{period}_{year_val}.jsonl"
            append_jsonl(str(articles_file), article)
            total_articles += 1
            total_records += 1

        log.info(f"    -> {total_articles} issues with war content so far")

    log.info(f"\n  Step 5 complete: {total_articles}/{total_records} issues have war articles")
    return total_articles
