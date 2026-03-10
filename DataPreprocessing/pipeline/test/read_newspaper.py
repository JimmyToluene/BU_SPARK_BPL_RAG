"""
BPL Newspaper Reader
─────────────────────
Read a specific day's newspaper content in a human-readable way.

Usage:
  python read_newspaper.py --date 1920-03-15
  python read_newspaper.py --date 1920-03-15 --paper "Boston Traveler"
  python read_newspaper.py --date 1920-03-15 --output day.txt
  python read_newspaper.py --list-dates
  python read_newspaper.py --list-dates --year 1920
"""

import json
import argparse
import textwrap
from pathlib import Path
from collections import defaultdict

DATASET_FILE = "./Boston_Daily_Traveler/bpl_clean_dataset.json"
LINE_WIDTH   = 80

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    print(f"Loading {path} ...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    print(f"Loaded {len(records):,} records.\n")
    return records

def divider(char="=", width=LINE_WIDTH) -> str:
    return char * width

def wrap(text: str, indent: int = 0) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=LINE_WIDTH, initial_indent=prefix,
                         subsequent_indent=prefix)

# ── Format one newspaper issue ────────────────────────────────────────────────

def format_issue(rec: dict, show_full_text: bool = True) -> str:
    lines = []

    lines.append(divider("="))
    lines.append(f"  {rec.get('newspaper', 'Unknown Newspaper').upper()}")
    lines.append(f"  {rec.get('issue_date', rec.get('date_start', '')[:10])}")
    lines.append(divider("="))

    # Metadata block
    lines.append("")
    lines.append(f"  Collection  : {', '.join(rec.get('collection', [rec.get('collection', '?')]) if isinstance(rec.get('collection'), list) else [rec.get('collection', '?')])}")
    lines.append(f"  Date        : {rec.get('date_start', '')[:10]}")
    lines.append(f"  Pages       : {rec.get('page_count', '?')}")
    lines.append(f"  Length      : {rec.get('char_count', 0):,} characters")
    lines.append(f"  Source      : {rec.get('source_url', '')}")
    lines.append("")
    lines.append(divider("-"))

    if show_full_text:
        lines.append("")
        lines.append("  FULL TEXT")
        lines.append("")

        text = rec.get("clean_text", "")

        # Split into paragraphs for readability
        # Newspaper text is one big blob — split on sentence endings
        # to create readable chunks of ~500 chars
        chunks = []
        current = []
        current_len = 0

        for sentence in text.replace(". ", ".|").split("|"):
            sentence = sentence.strip()
            if not sentence:
                continue
            current.append(sentence)
            current_len += len(sentence)
            if current_len > 400:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

        if current:
            chunks.append(" ".join(current))

        for chunk in chunks:
            lines.append(wrap(chunk, indent=2))
            lines.append("")

    else:
        # Preview only — first 600 chars
        preview = rec.get("clean_text", "")[:600]
        lines.append("")
        lines.append("  PREVIEW (first 600 chars)")
        lines.append("")
        lines.append(wrap(preview, indent=2))
        lines.append("  [... truncated. Use --full to see complete text ...]")
        lines.append("")

    lines.append(divider("="))
    return "\n".join(lines)

# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_read_date(records, date, paper, full, output) -> None:
    """Read all newspaper issues for a specific date."""

    # Filter by date
    matches = [
        r for r in records
        if r.get("date_start", "").startswith(date)
    ]

    # Optionally filter by newspaper title
    if paper:
        matches = [
            r for r in matches
            if paper.lower() in r.get("newspaper", "").lower()
        ]

    if not matches:
        print(f"No issues found for date: {date}")
        if paper:
            print(f"(filtered by newspaper: '{paper}')")

        # Suggest nearby dates
        all_dates = sorted(set(
            r.get("date_start", "")[:10]
            for r in records
            if r.get("date_start")
        ))
        nearby = [d for d in all_dates if abs((len(d) - len(date))) == 0
                  and d[:7] == date[:7]]
        if nearby:
            print(f"\nDates available in {date[:7]}:")
            for d in nearby[:10]:
                count = sum(1 for r in records if r.get("date_start", "").startswith(d))
                print(f"  {d}  ({count} issue{'s' if count > 1 else ''})")
        return

    print(f"Found {len(matches)} issue(s) for {date}")
    if paper:
        print(f"Newspaper filter: '{paper}'")
    print()

    output_lines = []

    for rec in matches:
        formatted = format_issue(rec, show_full_text=full)
        print(formatted)
        output_lines.append(formatted)

    # Save to file if requested
    if output:
        Path(output).write_text("\n".join(output_lines), encoding="utf-8")
        print(f"\nSaved to: {output}")


def cmd_list_dates(records, year) -> None:
    """List all available dates in the dataset."""

    # Group by year → month → dates
    by_year = defaultdict(lambda: defaultdict(list))

    for rec in records:
        date = rec.get("date_start", "")[:10]
        if not date:
            continue
        y, m = date[:4], date[5:7]
        if year and y != year:
            continue
        by_year[y][m].append(date)

    if not by_year:
        msg = f"No dates found"
        if year:
            msg += f" for year {year}"
        print(msg)
        return

    total_issues = sum(
        len(dates)
        for months in by_year.values()
        for dates in months.values()
    )

    print(divider("="))
    print(f"  AVAILABLE DATES IN DATASET")
    if year:
        print(f"  Year: {year}")
    print(f"  Total issues: {total_issues:,}")
    print(divider("="))

    for y in sorted(by_year.keys()):
        year_total = sum(len(d) for d in by_year[y].values())
        print(f"\n  {y}  ({year_total} issues)")
        print(f"  " + divider("-", width=40))
        for m in sorted(by_year[y].keys()):
            dates = sorted(set(by_year[y][m]))
            print(f"    {y}-{m}: {len(dates)} issues  "
                  f"[{dates[0][8:]} - {dates[-1][8:]}]")

    print()


def cmd_summary(records: list[dict]) -> None:
    """Print a summary of the dataset."""

    dates  = [r.get("date_start", "")[:10] for r in records if r.get("date_start")]
    papers = defaultdict(int)
    for r in records:
        papers[r.get("newspaper", "unknown")] += 1

    print(divider("="))
    print("  DATASET SUMMARY")
    print(divider("="))
    print(f"  Total issues     : {len(records):,}")
    print(f"  Date range       : {min(dates)} → {max(dates)}" if dates else "  No dates found")
    print(f"  Total text       : {sum(r.get('char_count',0) for r in records)/1e6:.1f} MB")
    print()
    print("  By newspaper:")
    for paper, count in sorted(papers.items(), key=lambda x: -x[1]):
        print(f"    {paper:<45} {count:>6,} issues")
    print(divider("="))

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BPL Newspaper Reader — read a specific day in human-readable format"
    )
    parser.add_argument("--date",        help="Date to read e.g. 1920-03-15")
    parser.add_argument("--paper",       help="Filter by newspaper name e.g. 'Boston Traveler'")
    parser.add_argument("--full",        action="store_true", help="Show full text (default: preview only)")
    parser.add_argument("--output",      help="Save output to a text file e.g. day.txt")
    parser.add_argument("--list-dates",  action="store_true", help="List all available dates")
    parser.add_argument("--year",        help="Filter list-dates by year e.g. 1920")
    parser.add_argument("--summary",     action="store_true", help="Show dataset summary")
    parser.add_argument("--dataset",     default=DATASET_FILE, help="Path to dataset JSON file")

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Dataset not found: {args.dataset}")
        print("Run the ingestion pipeline first.")
        return

    records = load_dataset(args.dataset)

    if args.summary:
        cmd_summary(records)
    elif args.list_dates:
        cmd_list_dates(records, year=args.year)
    elif args.date:
        cmd_read_date(
            records  = records,
            date     = args.date,
            paper    = args.paper,
            full     = args.full,
            output   = args.output,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python read_newspaper.py --summary")
        print("  python read_newspaper.py --list-dates")
        print("  python read_newspaper.py --list-dates --year 1920")
        print("  python read_newspaper.py --date 1920-03-15")
        print("  python read_newspaper.py --date 1920-03-15 --full")
        print("  python read_newspaper.py --date 1920-03-15 --output march15.txt")

if __name__ == "__main__":
    main()