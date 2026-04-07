"""
Shared utility functions for the pipeline and analysis scripts.
"""

import json
from pathlib import Path
from typing import Any

import yaml

from .constants import WORD_RE


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> Any:
    """Load a JSON file. Returns None if file does not exist."""
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path: str, data: Any) -> None:
    """Write data to a JSON file with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_records(data: Any) -> list[dict]:
    """
    Extract a records list from a JSON structure.
    Supports both ``{"records": [...]}`` dicts and raw ``[...]`` lists.
    """
    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported JSON structure: {type(data).__name__}")
    if not isinstance(records, list):
        raise ValueError("Expected `records` to be a list (or top-level list).")
    return records


def build_bln600_wordset(bln600_root, verbose: bool = False) -> set[str]:
    """
    Build a historical wordlist from BLN600 ground truth files.
    Uses ground truth only -- OCR text would pollute the wordlist with noise.
    """
    if isinstance(bln600_root, str):
        bln600_root = Path(bln600_root)

    gt_dir = bln600_root / "Ground Truth"
    if not gt_dir.exists():
        raise FileNotFoundError(f"BLN600 Ground Truth directory not found at: {gt_dir}")

    wordset: set[str] = set()
    files_loaded = 0

    for fp in gt_dir.iterdir():
        if fp.suffix.lower() != ".txt":
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
            words = WORD_RE.findall(text)
            wordset.update(w.lower() for w in words if len(w) > 2)
            files_loaded += 1
        except Exception as e:
            if verbose:
                print(f"  Warning: could not read {fp.name} -- {e}")

    if verbose:
        print(
            f"BLN600 wordlist built from {files_loaded} ground truth files "
            f"-- {len(wordset):,} unique words"
        )

    return wordset
