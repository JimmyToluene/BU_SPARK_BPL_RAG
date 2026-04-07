"""
OCR Evaluation Pipeline
=======================
Combines supervised metrics (CER, WER using BLN600 ground truth)
with unsupervised metrics (character-level heuristics, OOV rate).
Outputs a corpus-level aggregate report to console and JSON.

Install dependencies:
    pip install jiwer numpy wordfreq

Usage:
    from evaluate import evaluate_corpus

    report = evaluate_corpus(
        ocr_dir="BLN600/OCR Text",
        gt_dir="BLN600/Ground Truth",
        cleaned_dir="output/cleaned",   # optional — your cleaned text
        wordlist=wordlist,              # optional — from build_bln600_wordlist()
        output_path="report.json",
    )
"""

import json
import numpy as np
from pathlib import Path
from jiwer import cer, wer

from .metrics import (
    character_anomaly_rate,
    composite_score,
    non_alpha_token_rate,
    oov_rate,
    short_token_rate,
)


# ─────────────────────────────────────────────────────────────
# Supervised Metrics (require ground truth)
# ─────────────────────────────────────────────────────────────

def character_error_rate(hypothesis: str, reference: str) -> float:
    """
    CER — fraction of characters that are incorrect relative to ground truth.
    Lower is better. 0.0 = perfect match.
    """
    return round(cer(reference, hypothesis), 4)


def word_error_rate(hypothesis: str, reference: str) -> float:
    """
    WER — fraction of words that are incorrect relative to ground truth.
    Lower is better. 0.0 = perfect match.
    """
    return round(wer(reference, hypothesis), 4)


# ─────────────────────────────────────────────────────────────
# Per-Document Scorer
# ─────────────────────────────────────────────────────────────

def score_document(
    text: str,
    reference: str = None,
    wordlist: set = None,
) -> dict:
    """
    Compute all metrics for a single document.

    Args:
        text:      The text to evaluate (OCR or cleaned).
        reference: Ground truth text for supervised metrics (optional).
        wordlist:  Historical wordlist for OOV rate (optional).

    Returns:
        Dict of metric scores for this document.
    """
    ca  = character_anomaly_rate(text)
    na  = non_alpha_token_rate(text)
    st  = short_token_rate(text)
    cs  = composite_score(ca, na, st)
    oov = oov_rate(text, wordlist)

    result = {
        "composite_score":        cs,
        "character_anomaly_rate": ca,
        "non_alpha_token_rate":   na,
        "short_token_rate":       st,
        "oov_rate":               oov,
    }

    if reference is not None:
        result["cer"] = character_error_rate(text, reference)
        result["wer"] = word_error_rate(text, reference)

    return result


# ─────────────────────────────────────────────────────────────
# Corpus Evaluator
# ─────────────────────────────────────────────────────────────

def _aggregate(scores: list[dict]) -> dict:
    """Compute mean, std, and percentiles across all document scores."""
    if not scores:
        return {}

    keys = scores[0].keys()
    result = {}

    for key in keys:
        vals = [s[key] for s in scores if key in s]
        if not vals:
            continue
        result[key] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)), 4),
            "p10":  round(float(np.percentile(vals, 10)), 4),
            "p50":  round(float(np.percentile(vals, 50)), 4),
            "p90":  round(float(np.percentile(vals, 90)), 4),
        }

    return result


def evaluate_corpus(
    ocr_dir: str,
    gt_dir: str = None,
    cleaned_dir: str = None,
    wordlist: set = None,
    output_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate OCR quality across a corpus. Produces a corpus-level
    aggregate report comparing raw OCR against cleaned text (if provided).

    Args:
        ocr_dir:      Directory of raw OCR .txt files (indexed by GALE ID).
        gt_dir:       Directory of ground truth .txt files (optional, enables CER/WER).
        cleaned_dir:  Directory of cleaned .txt files (optional, for before/after comparison).
        wordlist:     Historical wordlist from build_bln600_wordlist() (optional).
        output_path:  Path to save JSON report (optional).
        verbose:      Print progress.

    Returns:
        Report dict with corpus-level aggregate metrics.
    """
    ocr_path     = Path(ocr_dir)
    gt_path      = Path(gt_dir) if gt_dir else None
    cleaned_path = Path(cleaned_dir) if cleaned_dir else None

    ocr_files = sorted(ocr_path.glob("*.txt"))

    if not ocr_files:
        raise FileNotFoundError(f"No .txt files found in {ocr_path}")

    if verbose:
        print(f"Evaluating {len(ocr_files)} documents...")

    ocr_scores     = []
    cleaned_scores = []
    matched        = 0
    skipped        = 0

    for ocr_file in ocr_files:
        doc_id = ocr_file.stem

        try:
            ocr_text = ocr_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            if verbose:
                print(f"  Skipping {doc_id} — could not read OCR file: {e}")
            skipped += 1
            continue

        # Load ground truth if available
        reference = None
        if gt_path:
            gt_file = gt_path / ocr_file.name
            if gt_file.exists():
                reference = gt_file.read_text(encoding="utf-8", errors="replace")

        # Score raw OCR
        ocr_scores.append(score_document(ocr_text, reference=reference, wordlist=wordlist))

        # Score cleaned text if provided
        if cleaned_path:
            cleaned_file = cleaned_path / ocr_file.name
            if cleaned_file.exists():
                cleaned_text = cleaned_file.read_text(encoding="utf-8", errors="replace")
                cleaned_scores.append(score_document(cleaned_text, reference=reference, wordlist=wordlist))

        matched += 1

    if verbose:
        print(f"  Matched: {matched} | Skipped: {skipped}")

    # Build report
    report = {
        "corpus": {
            "total_documents": len(ocr_files),
            "evaluated":       matched,
            "skipped":         skipped,
            "ground_truth":    gt_dir is not None,
            "cleaned_compared":cleaned_dir is not None,
        },
        "ocr_raw": _aggregate(ocr_scores),
    }

    if cleaned_scores:
        report["ocr_cleaned"] = _aggregate(cleaned_scores)

        # Delta — how much each metric improved after cleaning
        # Positive delta = improvement for composite/cer/wer (lower error, higher quality)
        raw_agg     = report["ocr_raw"]
        cleaned_agg = report["ocr_cleaned"]
        delta = {}

        for key in raw_agg:
            if key in cleaned_agg:
                raw_mean     = raw_agg[key]["mean"]
                cleaned_mean = cleaned_agg[key]["mean"]
                # For quality metrics (composite), higher is better
                # For error metrics (cer, wer, anomaly rates), lower is better
                if key == "composite_score":
                    delta[key] = round(cleaned_mean - raw_mean, 4)
                else:
                    delta[key] = round(raw_mean - cleaned_mean, 4)  # positive = improvement

        report["delta"] = delta

    # Print to console
    print("\n" + "="*60)
    print("OCR EVALUATION REPORT")
    print("="*60)
    print(json.dumps(report, indent=2))

    # Save to JSON
    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved to '{output_path}'")

    return report
