"""Shared utilities for the war-period pseudo GT pipeline."""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

log = logging.getLogger("war_gt")

# ── Shared HTTP session ──────────────────────────────────────────────────────

session = requests.Session()
session.headers.update({"User-Agent": "BPL-RAG-EvalPipeline/1.0"})

# ── Optional-dependency flags ────────────────────────────────────────────────

try:
    from PIL import Image          # noqa: F401
    import base64 as _base64       # noqa: F401

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from openai import OpenAI      # noqa: F401

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import torch                                        # noqa: F401
    from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# ── Config ───────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── JSON / JSONL helpers ─────────────────────────────────────────────────────


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


def read_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── HTTP with retry ──────────────────────────────────────────────────────────


def get_with_retry(url: str, retry_cfg: dict, timeout: int = 20,
                   is_text: bool = False):
    max_retries = retry_cfg.get("max_retries", 5)
    backoff = retry_cfg.get("backoff", [2, 5, 10, 30, 60])

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=timeout)
            if is_text and resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            wait = backoff[min(attempt, len(backoff) - 1)]
            log.warning(f"  Timeout {attempt + 1}/{max_retries}, retry in {wait}s")
            time.sleep(wait)
        except requests.exceptions.ConnectionError:
            wait = backoff[min(attempt, len(backoff) - 1)]
            log.warning(f"  ConnErr {attempt + 1}/{max_retries}, retry in {wait}s")
            time.sleep(wait)
        except requests.exceptions.HTTPError:
            if resp.status_code >= 500:
                wait = backoff[min(attempt, len(backoff) - 1)]
                log.warning(f"  HTTP {resp.status_code} {attempt + 1}/{max_retries}, retry in {wait}s")
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


# ── Text cleaning ────────────────────────────────────────────────────────────

REPLACEMENTS = {
    "|": "I", "\u00a6": "I", "\u00a9": "o", "\u00b0": "o",
    "\u017f": "s", "\u222b": "s", "\x00": "",
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
