"""
eda_token_cost.py – Per-page token estimation for embedding cost planning.

For each newspaper collection, randomly samples ONE issue per available year
(random day/month), estimates tokens for a single page, and produces a
summary report with visualisations.

Memory-safe: loads one year file at a time, keeps only the sampled record,
frees everything else immediately.

Usage:
    python eda_token_cost.py
    python eda_token_cost.py --seed 42
    python eda_token_cost.py --root /path/to/DataPreprocessing --encoding cl100k_base
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

YEAR_RE = re.compile(r"^\d{4}\.json$")

# Known data directories relative to project root.
# Each tuple: (collection_display_name, relative_path_to_year_files)
COLLECTIONS = [
    (
        "Boston_Evening_Transcript",
        Path("Boston_Evening_Transcript") / "bpl_clean_dataset",
    ),
    (
        "The_Boston_Traveler",
        Path("Boston_Traveller")
        / "Boston_Traveler_The_Boston_Traveler"
        / "The_Boston_Traveler",
    ),
]


# ── helpers ──────────────────────────────────────────────────────────────────


def discover_year_files(root: Path) -> dict[str, list[Path]]:
    """Return {collection_name: [sorted year-file paths]}."""
    found: dict[str, list[Path]] = {}
    for name, rel in COLLECTIONS:
        dpath = root / rel
        if not dpath.is_dir():
            continue
        files = sorted(f for f in dpath.iterdir() if YEAR_RE.match(f.name))
        if files:
            found[name] = files
    return found


def count_tokens(text: str, encoding=None) -> int:
    """Token count via tiktoken (if available) or word-based estimate."""
    if encoding is not None:
        return len(encoding.encode(text))
    # ~1.3 tokens per whitespace-delimited word is a common English estimate
    # for cl100k_base / o200k_base encodings.
    return int(len(text.split()) * 1.3)


def sample_one_record(path: Path, rng: random.Random) -> dict | None:
    """Load a year file, pick one random record, return essential fields only."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])
    elif isinstance(data, list):
        records = data
    else:
        return None

    if not records:
        return None

    rec = rng.choice(records)
    sampled = {
        "ark_id": rec.get("ark_id", ""),
        "newspaper": rec.get("newspaper", ""),
        "issue_date": rec.get("issue_date", ""),
        "date_iso": rec.get("date_iso", []),
        "page_count": rec.get("page_count", 1),
        "char_count": rec.get("char_count", 0),
        "clean_text": rec.get("clean_text", ""),
    }
    # Free the large structures right away.
    del data, records
    return sampled


def percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (n - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Estimate per-page token counts by sampling one issue per year "
            "for embedding-model cost estimation."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    ap.add_argument(
        "--encoding",
        default="cl100k_base",
        help=(
            "tiktoken encoding name (default: cl100k_base). "
            "Falls back to word-based estimate if tiktoken is not installed."
        ),
    )
    ap.add_argument(
        "--root",
        default=None,
        help="Project root directory (default: directory containing this script).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: analysis/token_cost_eda.json).",
    )
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    rng = random.Random(args.seed)

    # Try tiktoken; fall back gracefully.
    encoding = None
    encoding_name = "word_estimate_x1.3"
    try:
        import tiktoken

        encoding = tiktoken.get_encoding(args.encoding)
        encoding_name = args.encoding
        print(f"Using tiktoken encoding: {encoding_name}")
    except (ImportError, Exception):
        print(
            "tiktoken not available; using word-based token estimate (~1.3 tok/word)."
        )

    # Discover data.
    collections = discover_year_files(root)
    if not collections:
        sys.exit("No year-level JSON files found. Check --root path.")

    total_files = sum(len(v) for v in collections.values())
    print(f"Found {len(collections)} collection(s), {total_files} year files total.\n")

    # ── Sample one issue per year ────────────────────────────────────────────
    samples: list[dict] = []

    for coll_name, year_files in sorted(collections.items()):
        print(f"── {coll_name} ({len(year_files)} years) ──")
        for yf in year_files:
            year = yf.stem
            rec = sample_one_record(yf, rng)
            if rec is None:
                print(f"  {year}: no records, skipped")
                continue

            text = rec["clean_text"]
            page_count = max(rec["page_count"], 1)

            total_tokens = count_tokens(text, encoding)
            tokens_per_page = total_tokens / page_count
            chars_per_page = len(text) / page_count
            words = len(text.split())
            words_per_page = words / page_count

            sample = {
                "collection": coll_name,
                "year": int(year),
                "issue_date": rec["issue_date"],
                "date_iso": rec["date_iso"],
                "ark_id": rec["ark_id"],
                "page_count": page_count,
                "char_count_actual": len(text),
                "word_count": words,
                "total_tokens": total_tokens,
                "tokens_per_page": round(tokens_per_page, 1),
                "chars_per_page": round(chars_per_page, 1),
                "words_per_page": round(words_per_page, 1),
            }
            samples.append(sample)

            print(
                f"  {year} | {rec['issue_date']:30s} | {page_count:2d} pg | "
                f"{total_tokens:>7,} tok | {tokens_per_page:>7,.0f} tok/pg"
            )

            # Free text early.
            del text, rec

    if not samples:
        sys.exit("No records sampled.")

    # ── Aggregate statistics ─────────────────────────────────────────────────
    all_tpp = sorted(s["tokens_per_page"] for s in samples)
    n = len(all_tpp)

    total_pages_sampled = sum(s["page_count"] for s in samples)
    total_tokens_sampled = sum(s["total_tokens"] for s in samples)

    summary = {
        "num_collections": len(collections),
        "num_years_sampled": n,
        "total_pages_in_sampled_issues": total_pages_sampled,
        "total_tokens_in_sampled_issues": total_tokens_sampled,
        "tokens_per_page": {
            "mean": round(sum(all_tpp) / n, 1),
            "median": round(percentile(all_tpp, 50), 1),
            "p5": round(percentile(all_tpp, 5), 1),
            "p25": round(percentile(all_tpp, 25), 1),
            "p75": round(percentile(all_tpp, 75), 1),
            "p95": round(percentile(all_tpp, 95), 1),
            "min": round(min(all_tpp), 1),
            "max": round(max(all_tpp), 1),
        },
        "encoding": encoding_name,
        "seed": args.seed,
    }

    # ── Cost estimation ──────────────────────────────────────────────────────
    # Embedding model pricing (USD per 1M tokens, as of 2025).
    pricing = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    }

    mean_tpp = summary["tokens_per_page"]["mean"]
    cost_estimates: dict[str, dict] = {}
    for model, price_per_1m in pricing.items():
        cost_per_page = mean_tpp * price_per_1m / 1_000_000
        cost_estimates[model] = {
            "price_per_1M_tokens_usd": price_per_1m,
            "est_cost_per_page_usd": round(cost_per_page, 6),
            "est_cost_per_1000_pages_usd": round(cost_per_page * 1_000, 4),
            "est_cost_per_10000_pages_usd": round(cost_per_page * 10_000, 4),
        }

    report = {
        "summary": summary,
        "cost_estimates": cost_estimates,
        "samples": samples,
    }

    # ── Save JSON report ─────────────────────────────────────────────────────
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else analysis_dir / "token_cost_eda.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Console summary ──────────────────────────────────────────────────────
    tpp = summary["tokens_per_page"]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n} years sampled across {len(collections)} collection(s)")
    print(f"{'=' * 60}")
    print(f"  Tokens/page  mean:       {tpp['mean']:>8,.1f}")
    print(f"  Tokens/page  median:     {tpp['median']:>8,.1f}")
    print(f"  Tokens/page  p5 – p95:   {tpp['p5']:>8,.1f} – {tpp['p95']:>8,.1f}")
    print(f"  Tokens/page  min – max:  {tpp['min']:>8,.1f} – {tpp['max']:>8,.1f}")
    print()
    print("  Embedding cost estimates (per 1,000 pages):")
    for model, est in cost_estimates.items():
        print(f"    {model:30s}  ${est['est_cost_per_1000_pages_usd']:.4f}")
    print(f"\n  Report: {out_path}")

    # ── Visualisations ───────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    coll_names = sorted(set(s["collection"] for s in samples))
    colors = plt.cm.tab10.colors

    # 1) Tokens per page by year (scatter).
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cname in enumerate(coll_names):
        xs = [s["year"] for s in samples if s["collection"] == cname]
        ys = [s["tokens_per_page"] for s in samples if s["collection"] == cname]
        ax.scatter(
            xs, ys, label=cname, color=colors[i % len(colors)], alpha=0.7, s=30
        )
    ax.axhline(
        tpp["mean"],
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"mean = {tpp['mean']:,.0f}",
    )
    ax.axhline(
        tpp["median"],
        color="blue",
        linestyle=":",
        linewidth=1,
        label=f"median = {tpp['median']:,.0f}",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Tokens per page")
    ax.set_title("Tokens per Page by Year (1 sampled issue per year)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis_dir / "tokens_per_page_by_year.png", dpi=150)
    plt.close(fig)

    # 2) Histogram of tokens per page.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_tpp, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(
        tpp["mean"],
        color="red",
        linestyle="--",
        label=f"mean = {tpp['mean']:,.0f}",
    )
    ax.axvline(
        tpp["median"],
        color="blue",
        linestyle=":",
        label=f"median = {tpp['median']:,.0f}",
    )
    ax.set_xlabel("Tokens per page")
    ax.set_ylabel("Number of sampled issues")
    ax.set_title("Distribution of Tokens per Page")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis_dir / "tokens_per_page_hist.png", dpi=150)
    plt.close(fig)

    # 3) Page-count distribution.
    page_counts = [s["page_count"] for s in samples]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        page_counts,
        bins=range(min(page_counts), max(page_counts) + 2),
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlabel("Pages per issue")
    ax.set_ylabel("Number of sampled issues")
    ax.set_title("Distribution of Pages per Issue")
    fig.tight_layout()
    fig.savefig(analysis_dir / "pages_per_issue_hist.png", dpi=150)
    plt.close(fig)

    print(f"  Plots: {analysis_dir}/")


if __name__ == "__main__":
    main()
