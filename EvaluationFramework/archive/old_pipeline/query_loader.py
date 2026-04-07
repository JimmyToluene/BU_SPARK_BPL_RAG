"""Load and save test queries in JSONL format."""

import json
import logging
import os

log = logging.getLogger(__name__)


def load_queries(path: str) -> list[dict]:
    """Load test queries from a JSONL file.

    Args:
        path: Path to queries JSONL (e.g. ww1_ww2_test_queries.jsonl).

    Returns:
        List of query dicts with keys: question, question_type, ground_truths, answer, notes.
    """
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    log.info(f"Loaded {len(queries)} queries from {path}")
    return queries


def load_existing_output(path: str) -> dict[str, dict]:
    """Load already-processed results from a previous run (for resume support).

    Args:
        path: Path to output JSONL file.

    Returns:
        Dict mapping question text -> full output record.
    """
    existing = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    existing[entry["question"]] = entry
        log.info(f"Found {len(existing)} existing results in {path} (will skip)")
    return existing


def append_result(path: str, record: dict) -> None:
    """Append a single result record to the output JSONL file.

    Args:
        path: Path to output JSONL file.
        record: Dict to write as one JSON line.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
