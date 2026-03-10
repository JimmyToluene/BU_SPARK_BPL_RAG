# BPL Newspaper Data Ingestion Pipeline

Fetches historical Boston newspaper data (OCR text + metadata) from the [Digital Commonwealth](https://www.digitalcommonwealth.org) API and outputs structured JSON files.

## Requirements

- Python 3.9+
- `pip install requests pyyaml`

## Quick Start

```bash
cd DataPreprocessing

# Edit config.yaml to set your desired year range and newspapers
# Then run the full pipeline:
python main.py
```

## Pipeline Phases

| Phase | What it does | Command |
|-------|-------------|---------|
| 1 | Fetch record IDs from BPL API | `python main.py --phase 1` |
| 2 | Fetch OCR text + metadata for each record | `python main.py --phase 2` |
| 3 | Split year files into single-day files | `python main.py --phase 3` |

Run all phases: `python main.py`

### Test mode

Limits Phase 1 to 2 pages (~200 records) per collection:

```bash
python main.py --test
```

## Configuration (`config.yaml`)

```yaml
scope:
  date_start: "1920"      # Start year (API only supports year-level filtering)
  date_end:   "1920"      # End year

newspapers:
  - title: "Boston Evening Transcript"
    collections:
      - name: "Boston Evening Transcript"
        url: "https://www.digitalcommonwealth.org/search.json?..."

performance:
  max_workers: 4           # Parallel threads for Phase 2
  max_pages:   null        # Set to integer (e.g. 2) to limit during testing
```

## Output Structure

```
bpl_clean_dataset/
  Boston_Evening_Transcript_Boston_Evening_Transcript/
    bpl_ids.json                # Phase 1: record IDs
    1920.json                   # Phase 2: all records for year
    1920_by_single_day/         # Phase 3: one file per day
      1920-01-02.json
      1920-01-03.json
      ...
    temp/
      bpl_checkpoint.json       # Phase 2 resume checkpoint
```

### Record fields (in year/day JSON files)

| Field | Example |
|-------|---------|
| `ark_id` | `"9c681r286"` |
| `title` | `"Boston Evening Transcript"` |
| `issue_date` | `"January 06, 1920"` |
| `date_iso` | `["1920-01-06"]` |
| `clean_text` | Full cleaned OCR text |
| `char_count` | `206491` |
| `topics` | `["Local news"]` |
| `geography` | `["Boston", "Massachusetts"]` |
| `newspaper` | `"Boston Evening Transcript"` |
| `collection` | `"Boston Evening Transcript"` |

## Project Structure

```
DataPreprocessing/
  main.py                          # Entry point
  config.yaml                      # Configuration
  pipeline/
    __init__.py
    phase1_fetch_ids.py            # Phase 1: fetch record IDs
    phase2_fetch_text.py           # Phase 2: fetch text + metadata
    split_by_day.py                # Phase 3: split year data by day
    test/
      read_newspaper.py            # Utility to inspect output data
```

## Resume Support

Phase 2 saves checkpoints to `temp/bpl_checkpoint.json`. If interrupted, re-running `python main.py --phase 2` will resume from where it left off. Delete the checkpoint file to start over.
