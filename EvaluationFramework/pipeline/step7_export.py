"""Step 7 — Export final deduplicated Q&A dataset + summary report."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .shared import log, read_jsonl


def run(cfg: dict, periods: list[str]) -> int:
    log.info("=" * 60)
    log.info("STEP 7: Export Final JSONL + Summary")
    log.info("=" * 60)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    qa_path = data_dir / out["qa_output"]

    if not qa_path.exists():
        log.error(f"  No Q&A file at {qa_path}")
        return 0

    # Deduplicate
    seen = set()
    qa_pairs = []
    for obj in read_jsonl(str(qa_path)):
        q = obj.get("question", "")
        if q not in seen:
            seen.add(q)
            qa_pairs.append(obj)

    filtered = [
        qa for qa in qa_pairs
        if qa.get("period", "") in periods or not qa.get("period")
    ]

    # Write final output
    final_path = data_dir / "war_period_qa_final.jsonl"
    with open(final_path, "w", encoding="utf-8") as f:
        for qa in filtered:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # Compute stats
    by_period = defaultdict(int)
    by_type = defaultdict(int)
    by_confidence = defaultdict(int)
    by_source = defaultdict(int)

    for qa in filtered:
        by_period[qa.get("period", "unknown")] += 1
        by_type[qa.get("question_type", "unknown")] += 1
        by_confidence[qa.get("confidence", "unknown")] += 1
        by_source["llm" if "llm" in qa.get("notes", "") else "template"] += 1

    log.info(f"  Total Q&A pairs : {len(filtered)}")
    log.info(f"  Output          : {final_path}")
    log.info(f"  By period       : {dict(by_period)}")
    log.info(f"  By type         : {dict(by_type)}")
    log.info(f"  By confidence   : {dict(by_confidence)}")
    log.info(f"  By source       : {dict(by_source)}")

    # Human-readable summary
    summary_path = data_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("War-Period Pseudo Ground Truth -- Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated : {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Total     : {len(filtered)}\n\n")
        f.write(f"By period     : {dict(by_period)}\n")
        f.write(f"By type       : {dict(by_type)}\n")
        f.write(f"By confidence : {dict(by_confidence)}\n")
        f.write(f"By source     : {dict(by_source)}\n\n")
        f.write("Sample questions:\n")
        for qa in filtered[:5]:
            f.write(f"  Q: {qa['question'][:80]}\n")
            f.write(f"  A: {qa['answer'][:80]}\n")
            f.write(f"  GT: {qa['ground_truths']}\n\n")

    log.info(f"  Summary         : {summary_path}")
    return len(filtered)
