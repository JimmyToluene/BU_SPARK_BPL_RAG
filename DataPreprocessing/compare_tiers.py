import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from wordfreq import zipf_frequency


WORD_RE = re.compile(r"[A-Za-z]+")

# Mirrors `pipeline/ocr_eval.py`
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

JUNK_CHARS = re.compile(r"[|¦©°∫¬§±×÷¤¨¸˜]")

PUNCT_ONLY = re.compile(r"^[.,;:!?\-\"'()]+$")
REAL_SHORT_WORDS = {
    "a", "i", "o",
    "an", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "me", "my", "no", "of", "on", "or",
    "so", "to", "up", "us", "we",
    "mr", "dr", "st", "vs",
}


def load_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_records(data: Any) -> list[dict]:
    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported JSON structure: {type(data).__name__}")
    if not isinstance(records, list):
        raise ValueError("Expected `records` to be a list (or top-level list).")
    return records


def build_bln600_wordset(bln600_root: Path) -> set[str]:
    gt_dir = bln600_root / "Ground Truth"
    if not gt_dir.exists():
        raise FileNotFoundError(f"BLN600 Ground Truth directory not found at: {gt_dir}")

    wordset: set[str] = set()
    for fp in gt_dir.iterdir():
        if fp.suffix.lower() != ".txt":
            continue
        text = fp.read_text(encoding="utf-8", errors="replace")
        words = WORD_RE.findall(text)
        wordset.update(w.lower() for w in words if len(w) > 2)
    return wordset


def character_anomaly_rate(text: str) -> float:
    if not text:
        return 0.0
    return round(len(JUNK_CHARS.findall(text)) / len(text), 4)


def non_alpha_token_rate(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    candidates = [t for t in tokens if not PUNCT_ONLY.match(t)]
    if not candidates:
        return 0.0
    non_alpha = sum(1 for t in candidates if not re.search(r"[a-zA-Z]", t))
    return round(non_alpha / len(candidates), 4)


def short_token_rate(text: str) -> float:
    tokens = [t for t in text.split() if re.search(r"[a-zA-Z]", t)]
    if not tokens:
        return 0.0
    short = 0
    for t in tokens:
        letters = re.sub(r"[^a-zA-Z]", "", t)
        if len(letters) <= 2 and letters.lower() not in REAL_SHORT_WORDS:
            short += 1
    return round(short / len(tokens), 4)


def symbol_rate(text: str) -> float:
    """
    Non-junk symbol rate:
      - count characters that are non-alphanumeric and non-space,
      - but not part of OCR junk symbols.
    """
    if not text:
        return 0.0
    junk_set = set("|¦©°∫¬§±×÷¤¨¸˜")
    weird = 0
    for c in text:
        if c.isalnum() or c.isspace():
            continue
        if c in junk_set:
            continue
        weird += 1
    return round(weird / len(text), 4)


def oov_rate(text: str, wordlist: set[str] | None = None) -> float:
    words = text.split()
    if not words:
        return 0.0

    reference = set(HISTORICAL_WHITELIST)
    if wordlist:
        reference.update(wordlist)

    zipf_cache: dict[str, float] = {}

    def is_known(word: str) -> bool:
        cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", word).lower()
        if not cleaned or len(cleaned) <= 2:
            return True
        if cleaned in reference:
            return True
        if cleaned[0].isupper():
            return True  # skip proper nouns
        if cleaned not in zipf_cache:
            zipf_cache[cleaned] = zipf_frequency(cleaned, "en")
        return zipf_cache[cleaned] > 1.5

    oov = sum(1 for w in words if not is_known(w))
    return round(oov / len(words), 4)


def composite_score(char_anom: float, non_alpha: float, short_tok: float) -> float:
    noise = 0.40 * char_anom + 0.35 * non_alpha + 0.25 * short_tok
    return round(max(0.0, min(1.0, 1.0 - noise)), 4)


def mean_std(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(xs, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def safe_get_text(rec: dict, field: str) -> str:
    if not isinstance(rec, dict):
        return ""
    txt = rec.get(field, "")
    return txt if isinstance(txt, str) else ""


def plot_bar_means(agg: dict[str, dict[str, float]], metrics: list[str], out_png: Path) -> None:
    """
    agg[tier][metric] => mean
    """
    tiers = list(agg.keys())
    means = {t: [agg[t][m]["mean"] for m in metrics] for t in tiers}

    x = np.arange(len(metrics))
    width = 0.25

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.2), 5.5))
    for i, tier in enumerate(tiers):
        ax.bar(x + (i - len(tiers) / 2) * width + width / 2, means[tier], width=width, label=tier)

    ax.set_title("Tier comparison (mean rates)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, ha="center")
    ax.set_ylabel("Mean rate / score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Compare raw_text vs tier_1_cleaned/tier_2_cleaned/tier_3_cleaned using OCR quality metrics."
    )
    ap.add_argument(
        "--path",
        required=True,
        help="Path to JSON file (year or single-day) containing records.",
    )
    ap.add_argument(
        "--bln600_root",
        default="DataPreprocessing/BLN600",
        help="Path to BLN600 root folder (default: DataPreprocessing/BLN600).",
    )
    ap.add_argument(
        "--raw_field",
        default="raw_text",
        help="Field name for raw text (default: raw_text).",
    )
    ap.add_argument(
        "--tier1_field",
        default="tier_1_cleaned",
        help="Field name for tier1 text (default: tier_1_cleaned).",
    )
    ap.add_argument(
        "--tier2_field",
        default="tier_2_cleaned",
        help="Field name for tier2 text (default: tier_2_cleaned).",
    )
    ap.add_argument(
        "--tier3_field",
        default="tier_3_cleaned",
        help="Field name for tier3 text (default: tier_3_cleaned).",
    )
    ap.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional limit of records for faster debug runs.",
    )
    args = ap.parse_args()

    in_path = Path(args.path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    data = load_json_any(in_path)
    records = extract_records(data)
    if args.max_records is not None:
        records = records[: args.max_records]

    bln_root = Path(args.bln600_root)
    bln_words = build_bln600_wordset(bln_root)

    tiers_to_fields = [
        ("raw", args.raw_field),
        ("tier1", args.tier1_field),
        ("tier2", args.tier2_field),
        ("tier3", args.tier3_field),
    ]

    # Determine which tiers exist
    fields_present: list[tuple[str, str]] = []
    for tier_name, field in tiers_to_fields:
        # check any record has a non-empty string
        if any(isinstance(rec, dict) and isinstance(rec.get(field, ""), str) and rec.get(field, "") for rec in records):
            fields_present.append((tier_name, field))

    if not fields_present:
        raise SystemExit("No tiers/fields found with non-empty text in the provided JSON.")

    # Compute per-tier metric arrays
    metrics = ["oov_rate", "character_anomaly_rate", "symbol_rate", "non_alpha_token_rate", "short_token_rate", "composite_score"]
    tier_metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tier_meta: dict[str, dict[str, Any]] = {}

    for tier_name, field in fields_present:
        values = defaultdict(list)
        used_records = 0

        for rec in records:
            text = safe_get_text(rec, field)
            if not text:
                continue
            used_records += 1

            ca = character_anomaly_rate(text)
            na = non_alpha_token_rate(text)
            st = short_token_rate(text)
            oov = oov_rate(text, wordlist=bln_words)
            sym = symbol_rate(text)
            cs = composite_score(ca, na, st)

            values["oov_rate"].append(oov)
            values["character_anomaly_rate"].append(ca)
            values["symbol_rate"].append(sym)
            values["non_alpha_token_rate"].append(na)
            values["short_token_rate"].append(st)
            values["composite_score"].append(cs)

        tier_metric_values[tier_name] = values
        tier_meta[tier_name] = {"records_used": used_records, "text_field": field}

    # Aggregate
    report: dict[str, Any] = {
        "input_file": str(in_path),
        "bln600_root": str(bln_root),
        "records_total": len(records),
        "tiers": tier_meta,
        "aggregates": {},
    }

    for tier_name in tier_metric_values.keys():
        report["aggregates"][tier_name] = {}
        for metric in metrics:
            stats = mean_std(tier_metric_values[tier_name][metric])
            # Convert all metrics to percentage scale for easier reading/comparison.
            for stat_key in ("mean", "std", "p10", "p50", "p90"):
                if stat_key in stats:
                    stats[stat_key] = stats[stat_key] * 100.0
            report["aggregates"][tier_name][metric] = stats

    # Output folder
    out_dir = Path(__file__).resolve().parent / "analysis" / "comparison" / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "comparison_report.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Visualizations: one PNG per metric
    import matplotlib.pyplot as plt

    tiers = list(report["aggregates"].keys())
    for metric in metrics:
        means = [report["aggregates"][t][metric]["mean"] for t in tiers]

        fig = plt.figure(figsize=(max(8, len(tiers) * 1.0), 5.5))
        ax = fig.add_subplot(111)
        ax.bar(tiers, means)
        ax.set_title(f"Mean {metric} across records (percentage)")
        ax.set_ylabel("Mean value (%)")
        ax.set_xticklabels(tiers, rotation=0)
        # Label bars with mean values
        for i, v in enumerate(means):
            ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        out_png = out_dir / f"comparison_{metric}_means.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    print(f"Saved comparison report: {out_json}")
    print(f"Saved per-metric comparison images in: {out_dir}")


if __name__ == "__main__":
    main()

