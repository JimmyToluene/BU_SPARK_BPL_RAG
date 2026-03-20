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

import re
import json
import numpy as np
from pathlib import Path
from jiwer import cer, wer
from wordfreq import zipf_frequency


# ─────────────────────────────────────────────────────────────
# Historical Whitelist (for OOV rate)
# ─────────────────────────────────────────────────────────────

HISTORICAL_WHITELIST = {
    "ye", "thee", "thou", "thy", "thine",
    "hath", "doth", "hast", "wilt", "shalt",
    "mayst", "wouldst", "couldst", "shouldst", "mightst",
    "art", "wert", "hadst",
    "hither", "thither", "whither", "wherein", "whereof",
    "thereof", "hereof", "hereby", "therein",
    "methinks", "perchance", "haply", "belike", "forsooth",
    "nay", "aye", "yea", "verily", "prithee", "pray",
    "forthwith", "heretofore", "henceforth", "notwithstanding",
    "publick", "musick", "logick", "ethick", "rhetorick",
    "connexion", "inflexion", "reflexion", "compleat",
    "shew", "shewed", "shewn", "staid", "spake", "writ",
    "honour", "colour", "labour", "neighbour", "favour",
    "centre", "theatre", "fibre", "lustre", "sceptre",
    "recognise", "organise", "realise", "civilise",
    "defence", "offence", "pretence", "licence",
    "esq", "viz", "ibid", "idem", "supra", "infra",
    "aforesaid", "aforementioned", "heretofore",
    "whereas", "whereby", "whereupon", "whereunto",
    "mr", "mrs", "dr", "sr", "jr", "rev", "hon", "gen",
}


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
# Unsupervised Metrics (no ground truth needed)
# ─────────────────────────────────────────────────────────────

JUNK_CHARS  = re.compile(r"[|¦©°∫¬§±×÷¤¨¸˜]")
PUNCT_ONLY  = re.compile(r"^[.,;:!?\-\"'()]+$")
REAL_SHORT_WORDS = {
    "a", "i", "o",
    "an", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "me", "my", "no", "of", "on", "or",
    "so", "to", "up", "us", "we",
    "mr", "dr", "st", "vs",
}


def character_anomaly_rate(text: str) -> float:
    """Fraction of characters that are known OCR junk symbols."""
    if not text:
        return 0.0
    return round(len(JUNK_CHARS.findall(text)) / len(text), 4)


def non_alpha_token_rate(text: str) -> float:
    """Fraction of tokens containing no letters."""
    tokens = text.split()
    if not tokens:
        return 0.0
    candidates = [t for t in tokens if not PUNCT_ONLY.match(t)]
    if not candidates:
        return 0.0
    non_alpha = sum(1 for t in candidates if not re.search(r"[a-zA-Z]", t))
    return round(non_alpha / len(candidates), 4)


def short_token_rate(text: str) -> float:
    """Fraction of 1-2 char fragments that are not real short words."""
    tokens = [t for t in text.split() if re.search(r"[a-zA-Z]", t)]
    if not tokens:
        return 0.0
    short = sum(
        1 for t in tokens
        if len(re.sub(r"[^a-zA-Z]", "", t)) <= 2
        and re.sub(r"[^a-zA-Z]", "", t).lower() not in REAL_SHORT_WORDS
    )
    return round(short / len(tokens), 4)


def oov_rate(text: str, wordlist: set = None) -> float:
    """
    Fraction of tokens not in the reference wordlist.
    Uses historical whitelist + BLN600 wordlist + lowered zipf threshold
    to avoid penalising valid archaic vocabulary.
    """
    words = text.split()
    if not words:
        return 0.0

    reference = HISTORICAL_WHITELIST.copy()
    if wordlist:
        reference.update(wordlist)

    def is_known(word: str) -> bool:
        cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", word).lower()
        if not cleaned or len(cleaned) <= 2:
            return True
        if cleaned in reference:
            return True
        if cleaned[0].isupper():
            return True   # skip proper nouns
        return zipf_frequency(cleaned, "en") > 1.5

    oov = sum(1 for w in words if not is_known(w))
    return round(oov / len(words), 4)


def composite_score(
    char_anomaly: float,
    non_alpha: float,
    short_tok: float,
) -> float:
    """Single 0-1 quality score. Higher = cleaner."""
    noise = 0.40 * char_anomaly + 0.35 * non_alpha + 0.25 * short_tok
    return round(max(0.0, min(1.0, 1.0 - noise)), 4)


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