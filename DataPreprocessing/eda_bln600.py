import argparse
import json
import re
from collections import Counter
from pathlib import Path

from wordfreq import zipf_frequency


WORD_RE = re.compile(r"[A-Za-z]+")

# Matches the "OCR junk symbols" heuristic in `pipeline/ocr_eval.py`
OCR_JUNK_CHARS = set(list("|¦©°∫¬§±×÷¤¨¸˜"))

# The remaining constants mirror `pipeline/ocr_eval.py` so these rates
# are comparable with its evaluation logic.
PUNCT_ONLY = re.compile(r"^[.,;:!?\-\"'()]+$")
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
    # Denominator is total characters, matching pipeline/ocr_eval.py
    bad = sum(1 for c in text if c in OCR_JUNK_CHARS)
    return round(bad / len(text), 4)


def non_alpha_token_rate(text: str) -> float:
    """Fraction of non-punctuation tokens that contain no letters."""
    tokens = text.split()
    if not tokens:
        return 0.0
    candidates = [t for t in tokens if not PUNCT_ONLY.match(t)]
    if not candidates:
        return 0.0
    non_alpha = sum(1 for t in candidates if not re.search(r"[a-zA-Z]", t))
    return round(non_alpha / len(candidates), 4)


def short_token_rate(text: str) -> float:
    """
    Fraction of tokens that contain letters but reduce to <=2 letters after
    stripping non-letters, excluding a small list of real short words.
    """
    tokens = [t for t in text.split() if re.search(r"[a-zA-Z]", t)]
    if not tokens:
        return 0.0
    short = 0
    for t in tokens:
        letters = re.sub(r"[^a-zA-Z]", "", t)
        if len(letters) <= 2 and letters.lower() not in REAL_SHORT_WORDS:
            short += 1
    return round(short / len(tokens), 4)


def load_json_any(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])
        return records

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported JSON structure: top-level is {type(data).__name__}")


def build_bln600_wordset(bln600_root: Path) -> set[str]:
    """
    Build a BLN600 word set from `DataPreprocessing/BLN600/Ground Truth/*.txt`.
    Uses ground truth only (matches repository logic in `pipeline/tier2.py`).
    """
    gt_dir = bln600_root / "Ground Truth"
    if not gt_dir.exists():
        raise FileNotFoundError(f"BLN600 Ground Truth directory not found at: {gt_dir}")

    wordset: set[str] = set()
    for fp in gt_dir.iterdir():
        if fp.suffix.lower() != ".txt":
            continue

        text = fp.read_text(encoding="utf-8", errors="replace")
        words = WORD_RE.findall(text)
        # BLN600 wordlist in tier2 is lowercased and filters len>2
        wordset.update(w.lower() for w in words if len(w) > 2)

    return wordset


def char_categories(text: str):
    """
    Classify special characters.

    - "OCR junk": characters that are in OCR_JUNK_CHARS
    - "Other special": any non-alphanumeric and non-whitespace character that is not OCR junk
    """
    junk = []
    other = []
    for c in text:
        if c.isalnum() or c.isspace():
            continue
        if c in OCR_JUNK_CHARS:
            junk.append(c)
        else:
            other.append(c)
    return junk, other


def main():
    ap = argparse.ArgumentParser(
        description=(
            "EDA for BLN600 word membership and special characters in a JSON file "
            "(Phase 2/3 outputs)."
        )
    )
    ap.add_argument(
        "--path",
        required=True,
        help="Path to your year JSON file (e.g. 1900.json or 1930.json).",
    )
    ap.add_argument(
        "--field",
        default="clean_text",
        help="Record field containing text to analyze (default: clean_text).",
    )
    ap.add_argument(
        "--bln600_root",
        default="DataPreprocessing/BLN600",
        help="Path to BLN600 root folder (default: DataPreprocessing/BLN600).",
    )
    ap.add_argument(
        "--limit_records",
        type=int,
        default=None,
        help="Optional limit for faster runs.",
    )
    ap.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="How many special characters / words to show.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Path to write the JSON report to. "
            "Default: analysis/<input_stem>_<field>/<input_stem>_eda_bln600.json"
        ),
    )
    args = ap.parse_args()

    in_path = Path(args.path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    # Store all outputs under:
    #   analysis/<input_stem>_<field>/
    analysis_root = Path(__file__).resolve().parent / "analysis"
    analysis_dir = analysis_root / f"{in_path.stem}_{args.field}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load records from JSON
    records = load_json_any(in_path)
    if args.limit_records is not None:
        records = records[: args.limit_records]

    # Load BLN600 wordlist
    bln_root = Path(args.bln600_root)
    bln_words = build_bln600_wordset(bln_root)

    total_word_tokens = 0
    modern_word_tokens = 0
    bln_word_tokens = 0
    oov_word_tokens = 0

    unique_bln_words_found = set()
    unique_modern_words_found = set()
    unique_oov_words_found = set()
    unique_word_tokens = set()

    ocr_junk_char_counter = Counter()
    other_special_char_counter = Counter()

    # Cache wordfreq lookups for speed
    modern_cache: dict[str, bool] = {}

    def is_modern_word(w: str) -> bool:
        # wordfreq returns ~0 for unknown words; we treat > 0 as "exists"
        if w in modern_cache:
            return modern_cache[w]
        freq = zipf_frequency(w, "en")
        val = freq > 0.0
        modern_cache[w] = val
        return val

    # Iterate records
    num_records_used = 0
    num_records_missing_field = 0
    character_anomaly_rates: list[float] = []
    non_alpha_token_rates: list[float] = []
    short_token_rates: list[float] = []
    for rec in records:
        text = rec.get(args.field, "") if isinstance(rec, dict) else ""
        if not text:
            if isinstance(rec, dict) and args.field not in rec:
                num_records_missing_field += 1
            continue

        num_records_used += 1

        words = WORD_RE.findall(text)
        total_word_tokens += len(words)
        unique_word_tokens.update(w.lower() for w in words)

        for w in words:
            wl = w.lower()
            if is_modern_word(wl):
                modern_word_tokens += 1
                unique_modern_words_found.add(wl)
            elif wl in bln_words:
                bln_word_tokens += 1
                unique_bln_words_found.add(wl)
            else:
                oov_word_tokens += 1
                unique_oov_words_found.add(wl)

        # OCR evaluation rates (record-level)
        character_anomaly_rates.append(character_anomaly_rate(text))
        non_alpha_token_rates.append(non_alpha_token_rate(text))
        short_token_rates.append(short_token_rate(text))

        junk_chars, other_chars = char_categories(text)
        ocr_junk_char_counter.update(junk_chars)
        other_special_char_counter.update(other_chars)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Prepare report
    report = {
        "input_file": str(in_path),
        "field": args.field,
        "num_records_total": len(records),
        "num_records_used": num_records_used,
        "num_records_missing_field_or_empty": num_records_missing_field,
        "bln600_wordlist_size": len(bln_words),
        "total_word_tokens": total_word_tokens,
        "modern_word_tokens": modern_word_tokens,
        "modern_word_token_ratio": (modern_word_tokens / total_word_tokens) if total_word_tokens else None,
        "bln600_word_tokens": bln_word_tokens,
        "bln600_word_token_ratio": (bln_word_tokens / total_word_tokens) if total_word_tokens else None,
        "oov_word_tokens": oov_word_tokens,
        "oov_word_token_ratio": (oov_word_tokens / total_word_tokens) if total_word_tokens else None,
        "unique_word_tokens_count": len(unique_word_tokens),
        "unique_modern_words_found_count": len(unique_modern_words_found),
        "unique_bln_words_found_count": len(unique_bln_words_found),
        "unique_oov_words_found_count": len(unique_oov_words_found),
        "unique_bln_words_found_sample": sorted(unique_bln_words_found)[: min(50, len(unique_bln_words_found))],
        "character_anomaly_rate_mean": _mean(character_anomaly_rates),
        "non_alpha_token_rate_mean": _mean(non_alpha_token_rates),
        "short_token_rate_mean": _mean(short_token_rates),
        "ocr_junk_special_chars": {
            "unique_count": len(ocr_junk_char_counter),
            "total_occurrences": sum(ocr_junk_char_counter.values()),
            "unique_chars_sorted": sorted(ocr_junk_char_counter.keys()),
            "top": ocr_junk_char_counter.most_common(args.top_n),
        },
        "other_special_chars": {
            "unique_count": len(other_special_char_counter),
            "total_occurrences": sum(other_special_char_counter.values()),
            "unique_chars_sorted": sorted(other_special_char_counter.keys()),
            "top": other_special_char_counter.most_common(args.top_n),
        },
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = analysis_dir / (in_path.stem + "_eda_bln600.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\nEDA report saved to: {out_path}")

    # Visualizations
    # Keep optional: if matplotlib isn't installed, still export JSON.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping visualizations.")
        return

    def _bar_chart(items, title, xlabel, ylabel, output_file):
        # items: list[tuple[str, int]]
        if not items:
            return
        labels = [x for x, _ in items]
        counts = [y for _, y in items]

        fig = plt.figure(figsize=(max(8, len(items) * 0.6), 4.5))
        ax = fig.add_subplot(111)
        ax.bar(range(len(labels)), counts)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(labels)))
        # Keep special-character labels in their natural orientation.
        # (No diagonal rotation; avoids the "slanted" look.)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        fig.tight_layout()
        fig.savefig(output_file, dpi=150)
        plt.close(fig)

    # 1) BLN600 ratio
    ratio_modern = report["modern_word_token_ratio"]
    total_words = report["total_word_tokens"] if report["total_word_tokens"] else 0
    modern_pct = (100.0 * report["modern_word_tokens"] / total_words) if total_words else 0.0
    bln_pct = (100.0 * report["bln600_word_tokens"] / total_words) if total_words else 0.0
    oov_pct = (100.0 * report["oov_word_tokens"] / total_words) if total_words else 0.0
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    bins = ["Modern words", "BLN600 words", "OOV words"]
    values = [modern_pct, bln_pct, oov_pct]
    bars = ax.bar(bins, values)
    # Show percentage label above each bin
    for bar in bars:
        height = bar.get_height()
        # If BLN600 is tiny, 2 decimals can round to 0.00%.
        # Use higher precision for small percentages.
        fmt = "{:.4f}%" if height < 1.0 else "{:.2f}%"
        label = fmt.format(height)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Word Token Share: Modern vs BLN600 vs OOV")
    ax.set_ylabel("Percentage of word tokens (%)")
    fig.tight_layout()
    fig.savefig(analysis_dir / "bln600_token_share.png", dpi=150)
    plt.close(fig)

    # 1b) Token counts visualization (helps when BLN600% is extremely small)
    modern_count = int(report["modern_word_tokens"])
    bln_count = int(report["bln600_word_tokens"])
    oov_count = int(report["oov_word_tokens"])
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    bins = ["Modern words", "BLN600 words", "OOV words"]
    counts = [modern_count, bln_count, oov_count]
    bars = ax.bar(bins, counts)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{c:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Word Token Counts: Modern vs BLN600 vs OOV")
    ax.set_ylabel("Token count")
    fig.tight_layout()
    fig.savefig(analysis_dir / "bln600_token_share_counts.png", dpi=150)
    plt.close(fig)

    # 3) OCR quality rate visualizations
    rate_means = {
        "character_anomaly_rate": _mean(character_anomaly_rates),
        "non_alpha_token_rate": _mean(non_alpha_token_rates),
        "short_token_rate": _mean(short_token_rates),
    }

    # Mean bar chart
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111)
    rate_names = list(rate_means.keys())
    rate_vals = [rate_means[k] for k in rate_names]
    bars = ax.bar(rate_names, rate_vals)
    ax.set_title("OCR Rates (Mean) across Records")
    ax.set_ylabel("Rate (fraction)")
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(analysis_dir / "ocr_rates_mean_bar.png", dpi=150)
    plt.close(fig)

    # Histograms
    def _hist(values: list[float], title: str, output_file: Path, bins: int = 30):
        if not values:
            return
        fig = plt.figure(figsize=(6.5, 4.5))
        ax = fig.add_subplot(111)
        ax.hist(values, bins=bins)
        ax.set_title(title)
        ax.set_xlabel("Rate (fraction)")
        ax.set_ylabel("Number of records")
        fig.tight_layout()
        fig.savefig(output_file, dpi=150)
        plt.close(fig)

    _hist(
        character_anomaly_rates,
        "character_anomaly_rate distribution",
        analysis_dir / "character_anomaly_rate_hist.png",
    )
    _hist(
        non_alpha_token_rates,
        "non_alpha_token_rate distribution",
        analysis_dir / "non_alpha_token_rate_hist.png",
    )
    _hist(
        short_token_rates,
        "short_token_rate distribution",
        analysis_dir / "short_token_rate_hist.png",
    )

    # 2) OCR junk chars top N
    _bar_chart(
        report["ocr_junk_special_chars"]["top"],
        title="Top OCR-junk special characters (by occurrence)",
        xlabel="Special character",
        ylabel="Occurrences",
        output_file=str(analysis_dir / "top_ocr_junk_chars.png"),
    )

    # 3) Other special chars top N
    _bar_chart(
        report["other_special_chars"]["top"],
        title="Top other special characters (by occurrence)",
        xlabel="Special character",
        ylabel="Occurrences",
        output_file=str(analysis_dir / "top_other_special_chars.png"),
    )

    print(f"Visualizations saved to: {analysis_dir}")


if __name__ == "__main__":
    main()

