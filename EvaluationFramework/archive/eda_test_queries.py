#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) for test_queries.jsonl
======================================================
Produces a detailed console report and saves summary tables to
eda_report_test_queries.txt in the same directory.
"""

import json
import re
import os
from collections import Counter
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "test_queries.jsonl"
REPORT_PATH = SCRIPT_DIR / "eda_report_test_queries.txt"

records = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [WARN] Skipping line {lineno}: {e}")

N = len(records)

# ── Helper ───────────────────────────────────────────────────────────────────

def section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n  {title}\n{bar}"

def subsection(title: str) -> str:
    return f"\n--- {title} {'-' * max(0, 64 - len(title))}"

# ── Analysis ─────────────────────────────────────────────────────────────────

lines: list[str] = []

# 1. Overview
lines.append(section("1. OVERVIEW"))
lines.append(f"File            : {INPUT_PATH.name}")
lines.append(f"Total records   : {N}")

fields_present = set()
for r in records:
    fields_present.update(r.keys())
lines.append(f"Fields detected : {sorted(fields_present)}")

# 2. Field completeness
lines.append(section("2. FIELD COMPLETENESS"))
lines.append(f"{'Field':<20} {'Non-null':>10} {'Null':>10} {'% Complete':>12}")
lines.append("-" * 55)
for field in ["question", "question_type", "ground_truths", "answer", "notes"]:
    non_null = sum(1 for r in records if r.get(field) is not None)
    null_ct = N - non_null
    pct = non_null / N * 100 if N else 0
    lines.append(f"{field:<20} {non_null:>10} {null_ct:>10} {pct:>11.1f}%")

# 3. Question type distribution
lines.append(section("3. QUESTION TYPE DISTRIBUTION"))
qt_counter = Counter(r.get("question_type") for r in records)
lines.append(f"{'question_type':<20} {'Count':>8} {'%':>8}")
lines.append("-" * 38)
for qt, cnt in qt_counter.most_common():
    label = str(qt) if qt is not None else "<null>"
    lines.append(f"{label:<20} {cnt:>8} {cnt / N * 100:>7.1f}%")

# 4. Ground truths analysis
lines.append(section("4. GROUND TRUTHS ANALYSIS"))

gt_null = [i for i, r in enumerate(records) if r.get("ground_truths") is None]
gt_empty = [i for i, r in enumerate(records)
            if r.get("ground_truths") is not None and len(r["ground_truths"]) == 0]
gt_populated = [i for i, r in enumerate(records)
                if r.get("ground_truths") is not None and len(r["ground_truths"]) > 0]

lines.append(f"ground_truths == null    : {len(gt_null):>3}  (lines {[i+1 for i in gt_null]})")
lines.append(f"ground_truths == []      : {len(gt_empty):>3}  (lines {[i+1 for i in gt_empty]})")
lines.append(f"ground_truths populated  : {len(gt_populated):>3}  (lines {[i+1 for i in gt_populated]})")

if gt_populated:
    gt_sizes = [len(records[i]["ground_truths"]) for i in gt_populated]
    lines.append(f"\nGT counts among populated records:")
    lines.append(f"  min = {min(gt_sizes)}, max = {max(gt_sizes)}, "
                 f"mean = {sum(gt_sizes)/len(gt_sizes):.1f}")
    size_dist = Counter(gt_sizes)
    for sz, cnt in sorted(size_dist.items()):
        lines.append(f"  {sz} ground truths: {cnt} records")

# Unique ark_ids
all_ark_ids = []
for r in records:
    if r.get("ground_truths"):
        for gt in r["ground_truths"]:
            all_ark_ids.append(gt.get("ark_id", ""))
lines.append(f"\nTotal ark_id references : {len(all_ark_ids)}")
lines.append(f"Unique ark_ids          : {len(set(all_ark_ids))}")
dupes = {k: v for k, v in Counter(all_ark_ids).items() if v > 1}
if dupes:
    lines.append(f"Duplicate ark_ids       : {len(dupes)}")
    for ark, cnt in dupes.items():
        lines.append(f"  {ark} (appears {cnt}x)")
else:
    lines.append("Duplicate ark_ids       : 0")

# 5. Annotation quality tiers
lines.append(section("5. ANNOTATION QUALITY TIERS"))
tier_full = []      # question_type + non-empty GT
tier_gt_only = []   # GT but no question_type
tier_qt_only = []   # question_type but empty/null GT
tier_minimal = []   # neither

for i, r in enumerate(records):
    has_qt = r.get("question_type") is not None
    gt = r.get("ground_truths")
    has_gt = gt is not None and len(gt) > 0
    if has_qt and has_gt:
        tier_full.append(i + 1)
    elif has_gt:
        tier_gt_only.append(i + 1)
    elif has_qt:
        tier_qt_only.append(i + 1)
    else:
        tier_minimal.append(i + 1)

lines.append(f"{'Tier':<40} {'Count':>6}  Lines")
lines.append("-" * 70)
lines.append(f"{'Full (question_type + GT)':<40} {len(tier_full):>6}  {tier_full}")
lines.append(f"{'GT only (no question_type)':<40} {len(tier_gt_only):>6}  {tier_gt_only}")
lines.append(f"{'question_type only (empty/null GT)':<40} {len(tier_qt_only):>6}  {tier_qt_only}")
lines.append(f"{'Minimal (answer-only)':<40} {len(tier_minimal):>6}  {tier_minimal}")

# 6. Question text statistics
lines.append(section("6. QUESTION TEXT STATISTICS"))
q_lengths_words = [len(r["question"].split()) for r in records]
q_lengths_chars = [len(r["question"]) for r in records]
lines.append(f"Word count  — min: {min(q_lengths_words)}, max: {max(q_lengths_words)}, "
             f"mean: {sum(q_lengths_words)/N:.1f}")
lines.append(f"Char count  — min: {min(q_lengths_chars)}, max: {max(q_lengths_chars)}, "
             f"mean: {sum(q_lengths_chars)/N:.1f}")

# Question starting words
first_words = Counter(r["question"].split()[0].lower() for r in records)
lines.append(f"\nMost common first word:")
for word, cnt in first_words.most_common(5):
    lines.append(f"  '{word}' — {cnt}")

# 7. Answer statistics
lines.append(section("7. ANSWER TEXT STATISTICS"))
ans_lengths_words = [len(r["answer"].split()) for r in records]
ans_lengths_chars = [len(r["answer"]) for r in records]
lines.append(f"Word count  — min: {min(ans_lengths_words)}, max: {max(ans_lengths_words)}, "
             f"mean: {sum(ans_lengths_words)/N:.1f}")
lines.append(f"Char count  — min: {min(ans_lengths_chars)}, max: {max(ans_lengths_chars)}, "
             f"mean: {sum(ans_lengths_chars)/N:.1f}")

# 8. Notes field
lines.append(section("8. NOTES FIELD"))
notes_non_null = [(i + 1, r["notes"]) for i, r in enumerate(records)
                  if r.get("notes") is not None]
lines.append(f"Records with notes: {len(notes_non_null)} / {N}")
for lineno, note in notes_non_null:
    lines.append(f"  Line {lineno}: {note}")

# 9. Topical / temporal classification
lines.append(section("9. TOPICAL / TEMPORAL CLASSIFICATION"))

# keyword-based rough classification
era_keywords = {
    "Revolutionary War / Colonial": [
        "revolution", "colonial", "patriot", "1770", "1775", "1776",
        "bunker hill", "boston massacre", "tea party", "sons of liberty",
        "faneuil", "paul revere", "samuel adams", "crispus attucks",
        "old state house", "declaration of independence",
    ],
    "Civil War / Abolition (1830s-1860s)": [
        "abolitionist", "abolition", "slavery", "civil war",
        "frederick douglass", "liberator", "anti-slavery", "emancipation",
        "phillis wheatley",
    ],
    "WWI era (1914-1918)": [
        "world war i", "wwi", "great war", "1914", "1915", "1916",
        "1917", "1918", "trench",
    ],
    "Interwar (1919-1938)": [
        "1919", "prohibition", "great depression", "1929", "1930s",
    ],
    "WWII era (1939-1945)": [
        "world war ii", "wwii", "1939", "1940", "1941", "1942",
        "1943", "1944", "1945", "pearl harbor", "d-day", "nazi",
        "holocaust",
    ],
    "Post-1945 / Modern": [
        "jfk", "kennedy", "big dig", "marathon bombing", "2013",
    ],
    "Boston landmarks / general history": [
        "boston common", "beacon hill", "navy yard", "boston public library",
        "cape cod", "postcards",
    ],
    "Other / Non-Boston": [
        "indigenous", "zulu", "south africa",
    ],
}

classified: dict[str, list[int]] = {era: [] for era in era_keywords}
unclassified: list[int] = []

for i, r in enumerate(records):
    q_lower = r["question"].lower()
    a_lower = r["answer"].lower()
    text = q_lower + " " + a_lower
    matched = False
    for era, kws in era_keywords.items():
        if any(kw in text for kw in kws):
            classified[era].append(i + 1)
            matched = True
            break  # first match wins
    if not matched:
        unclassified.append(i + 1)

lines.append(f"{'Era / Topic':<40} {'Count':>6}  Lines")
lines.append("-" * 70)
for era, line_nums in classified.items():
    if line_nums:
        lines.append(f"{era:<40} {len(line_nums):>6}  {line_nums}")
if unclassified:
    lines.append(f"{'Unclassified':<40} {len(unclassified):>6}  {unclassified}")

# 10. War-period suitability
lines.append(section("10. WAR-PERIOD PSEUDO-GT SUITABILITY"))
lines.append(
    "Criteria: query topic aligns with WWI (1914-1918) or WWII (1939-1945)\n"
    "          AND ground_truths is non-null and non-empty.\n"
)

wwi_lines = classified.get("WWI era (1914-1918)", [])
wwii_lines = classified.get("WWII era (1939-1945)", [])
interwar_lines = classified.get("Interwar (1919-1938)", [])

war_related = sorted(set(wwi_lines + wwii_lines + interwar_lines))

lines.append(f"WWI-era queries          : {len(wwi_lines)}  {wwi_lines}")
lines.append(f"WWII-era queries         : {len(wwii_lines)}  {wwii_lines}")
lines.append(f"Interwar queries         : {len(interwar_lines)}  {interwar_lines}")
lines.append(f"Combined war-adjacent    : {len(war_related)}  {war_related}")

war_with_gt = [ln for ln in war_related if ln - 1 in gt_populated]
lines.append(f"\nWith populated GT        : {len(war_with_gt)}  {war_with_gt}")
lines.append(f"Without GT (need new GT) : {len(war_related) - len(war_with_gt)}")

lines.append(subsection("CONCLUSION"))
lines.append(
    f"Out of {N} total queries, {len(war_with_gt)} have both war-period relevance\n"
    f"and populated ground truths. The test set is heavily weighted toward\n"
    f"Revolutionary War / Colonial era ({len(classified.get('Revolutionary War / Colonial', []))} queries) "
    f"and Abolition ({len(classified.get('Civil War / Abolition (1830s-1860s)', []))} queries).\n"
    f"To evaluate a RAG system over 1900-1946 newspaper data for the two\n"
    f"World War periods, new purpose-built queries are needed."
)

# 11. Summary matrix
lines.append(section("11. PER-LINE SUMMARY MATRIX"))
lines.append(f"{'#':>3}  {'q_type':<13} {'GT_ct':>5}  {'ans_words':>9}  {'notes':>5}  Question (first 60 chars)")
lines.append("-" * 105)
for i, r in enumerate(records):
    qt = r.get("question_type") or "—"
    gt = r.get("ground_truths")
    gt_ct = len(gt) if gt is not None else "null"
    ans_w = len(r["answer"].split())
    has_notes = "Y" if r.get("notes") else "—"
    q_trunc = r["question"][:60]
    lines.append(f"{i+1:>3}  {qt:<13} {str(gt_ct):>5}  {ans_w:>9}  {has_notes:>5}  {q_trunc}")

# ── Output ───────────────────────────────────────────────────────────────────

report = "\n".join(lines)
print(report)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report + "\n")

print(f"\n{'=' * 70}")
print(f"  Report saved to: {REPORT_PATH}")
print(f"{'=' * 70}")
