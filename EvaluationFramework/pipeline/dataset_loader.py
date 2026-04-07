"""Load the cleaned BPL newspaper dataset from JSONL."""

import json
import logging

log = logging.getLogger(__name__)


def load_dataset(path: str) -> list[dict]:
    """Load the cleaned newspaper dataset (one JSON object per line).

    Args:
        path: Path to the JSONL file (e.g. bpl_clean_dataset.jsonl).

    Returns:
        List of record dicts with keys like ark_id, newspaper, clean_text, etc.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"Loaded {len(records)} records from {path}")
    return records
