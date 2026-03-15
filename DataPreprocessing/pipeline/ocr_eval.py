"""
Character-Level OCR Evaluation for Historical (1800s) Documents
================================================================
Three period-agnostic heuristics that require no dictionary or LM.

Install:
    pip install numpy

Usage:
    python ocr_evaluation_pipeline.py records.json
    python ocr_evaluation_pipeline.py records.json --field raw_text --limit 500
"""

import re
import json
import argparse
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Metric 1: Character Anomaly Rate
# Fraction of characters that are known OCR junk symbols.
# These never appear in clean English text of any era.
# ─────────────────────────────────────────────────────────────

JUNK_CHARS = re.compile(r"[|¦©°∫¬§±×÷¤¨¸˜]")

def character_anomaly_rate(text: str) -> float:
    if not text:
        return 0.0
    return len(JUNK_CHARS.findall(text)) / len(text)


# ─────────────────────────────────────────────────────────────
# Metric 2: Non-Alpha Token Rate
# Fraction of tokens that contain no letters at all (e.g. "||", "3|7").
# Pure punctuation tokens are excluded to avoid false positives.
# ─────────────────────────────────────────────────────────────

PUNCT_ONLY = re.compile(r"^[.,;:!?\-\"'()]+$")

def non_alpha_token_rate(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    candidates = [t for t in tokens if not PUNCT_ONLY.match(t)]
    if not candidates:
        return 0.0
    non_alpha = sum(1 for t in candidates if not re.search(r"[a-zA-Z]", t))
    return non_alpha / len(candidates)


# ─────────────────────────────────────────────────────────────
# Metric 3: Short Token Rate
# Fraction of tokens that are 1-2 letters long but NOT real short
# words. OCR errors frequently produce these as character fragments.
# ─────────────────────────────────────────────────────────────

REAL_SHORT_WORDS = {
    "a", "i", "o",
    "an", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "me", "my", "no", "of", "on", "or",
    "so", "to", "up", "us", "we",
    "mr", "dr", "st", "vs",
}

def short_token_rate(text: str) -> float:
    tokens = [t for t in text.split() if re.search(r"[a-zA-Z]", t)]
    if not tokens:
        return 0.0
    short = sum(
        1 for t in tokens
        if len(re.sub(r"[^a-zA-Z]", "", t)) <= 2
        and re.sub(r"[^a-zA-Z]", "", t).lower() not in REAL_SHORT_WORDS
    )
    return short / len(tokens)


# ─────────────────────────────────────────────────────────────
# Composite Score  (0 = very noisy, 1 = clean)
# All three metrics are noise indicators (lower = better).
# ─────────────────────────────────────────────────────────────

def composite_score(char_anomaly: float, non_alpha: float, short_tok: float) -> float:
    noise = 0.40 * char_anomaly + 0.35 * non_alpha + 0.25 * short_tok
    return round(max(0.0, min(1.0, 1.0 - noise)), 4)


# ─────────────────────────────────────────────────────────────
# Corpus Evaluator
# ─────────────────────────────────────────────────────────────

def evaluate_ocr(lines: list[str]) -> dict:
    ca_scores, na_scores, st_scores, cs_scores = [], [], [], []

    for line in lines:
        ca = character_anomaly_rate(line)
        na = non_alpha_token_rate(line)
        st = short_token_rate(line)
        cs = composite_score(ca, na, st)
        ca_scores.append(ca)
        na_scores.append(na)
        st_scores.append(st)
        cs_scores.append(cs)

    mean_cs = float(np.mean(cs_scores))

    if mean_cs >= 0.85:
        quality_band = "clean"
    elif mean_cs >= 0.65:
        quality_band = "moderate noise"
    elif mean_cs >= 0.40:
        quality_band = "significant noise"
    else:
        quality_band = "heavily degraded"

    return {
        "num_lines":                  len(lines),
        "quality_band":               quality_band,
        "composite_score_mean":       round(mean_cs, 4),
        "composite_score_std":        round(float(np.std(cs_scores)), 4),
        "char_anomaly_rate_mean":     round(float(np.mean(ca_scores)), 4),
        "non_alpha_token_rate_mean":  round(float(np.mean(na_scores)), 4),
        "short_token_rate_mean":      round(float(np.mean(st_scores)), 4),
        "composite_score_percentiles": {
            "p10": round(float(np.percentile(cs_scores, 10)), 4),
            "p25": round(float(np.percentile(cs_scores, 25)), 4),
            "p50": round(float(np.percentile(cs_scores, 50)), 4),
            "p75": round(float(np.percentile(cs_scores, 75)), 4),
            "p90": round(float(np.percentile(cs_scores, 90)), 4),
        },
        "tier_breakdown": {
            "clean_pct":    round(float(np.mean([s >= 0.85 for s in cs_scores])), 4),
            "moderate_pct": round(float(np.mean([0.65 <= s < 0.85 for s in cs_scores])), 4),
            "severe_pct":   round(float(np.mean([s < 0.65 for s in cs_scores])), 4),
        },
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Character-level OCR quality evaluation for historical documents."
    )
    parser.add_argument("path", help="Path to input JSON file")
    parser.add_argument("--field",  default="clean_text", help="Text field to evaluate (default: clean_text)")
    parser.add_argument("--limit",  type=int,             help="Cap number of records for faster evaluation")
    parser.add_argument("--output", "-o",                 help="Save JSON report to this path")
    args = parser.parse_args()

    in_path = Path(args.path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    with in_path.open(encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", data) if isinstance(data, dict) else data
    if args.limit:
        records = records[: args.limit]

    lines = [str(rec.get(args.field, "")) for rec in records if rec.get(args.field)]
    if not lines:
        raise SystemExit(f"No non-empty '{args.field}' fields found in {in_path}.")

    print(f"Evaluating {len(lines)} records...\n")
    report = evaluate_ocr(lines)
    output_str = json.dumps(report, indent=2)
    print(output_str)

    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
        print(f"\nReport saved to '{args.output}'")