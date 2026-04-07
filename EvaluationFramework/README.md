# BPL RAG Evaluation Framework

Builds **war-period pseudo ground truth** Q&A pairs from Boston Public Library newspaper archives (1914-1918, 1939-1945) for RAG evaluation.

Uses **dots.ocr** (3B VLM) for layout-aware OCR on historical newspaper scans, replacing the legacy Tesseract + multi-tier cleaning chain.

## Setup

```bash
pip install requests pyyaml Pillow openai
```

**For Step 4 (dots.ocr)** -- pick one mode:

```bash
# Option A: vLLM server (recommended, needs CUDA GPU)
pip install vllm
vllm serve rednote-hilab/dots.ocr.1.5-3B --dtype bfloat16 --max-model-len 8192

# Option B: In-process transformers
pip install transformers torch accelerate
```

**For Step 6 (LLM Q&A generation):**
```bash
export OPENAI_API_KEY="sk-..."
```

## Quick Start

```bash
cd EvaluationFramework

# Full pipeline (all 7 steps, both war periods)
python main.py

# Quick test: 5 records/year, skip image download + OCR
python main.py --sample 5 --steps 1 2 5 6 7

# WWII only
python main.py --period wwii

# Just fetch data (no GPU needed)
python main.py --steps 1 2
```

## Pipeline Steps

| Step | File | What it does |
|------|------|-------------|
| 1 | `step1_fetch_ids.py` | Fetch record IDs per year from Digital Commonwealth API |
| 2 | `step2_fetch_metadata.py` | Fetch metadata + API OCR text (parallel, checkpointed) |
| 3 | `step3_download_pages.py` | Download newspaper page images via IIIF manifests |
| 4 | `step4_dots_ocr.py` | Run dots.ocr VLM for layout-aware OCR |
| 5 | `step5_extract_articles.py` | Clean text, score segments by war-keyword density |
| 6 | `step6_generate_qa.py` | Generate Q&A pairs (GPT-4o or template fallback) |
| 7 | `step7_export.py` | Deduplicate, export final JSONL + summary |

Steps 1-2 need only `requests`. Steps 3-4 need a GPU. Steps 5-7 work on whatever OCR text is available (dots.ocr or API fallback).

## Project Structure

```
EvaluationFramework/
  main.py                     # CLI entry point
  war_period_config.yaml      # Pipeline configuration
  pipeline/
    __init__.py               # Step registry
    shared.py                 # Utilities, HTTP, checkpoint, text cleaning
    step1_fetch_ids.py        # Step 1
    step2_fetch_metadata.py   # Step 2
    step3_download_pages.py   # Step 3
    step4_dots_ocr.py         # Step 4 (dots.ocr)
    step5_extract_articles.py # Step 5
    step6_generate_qa.py      # Step 6
    step7_export.py           # Step 7
  test_queries.jsonl          # 38 general test queries (reference)
  eda_test_queries.py         # EDA script for test queries
  archive/                    # Legacy pipeline (pre-dots.ocr)
```

## Output

```
war_period_data/
  ids/                        # Per-year record ID files
  records/                    # Per-year metadata + API OCR text
  pages/                      # Downloaded IIIF page images
  ocr_dots/                   # dots.ocr output text
  articles/                   # Extracted war-related segments
  war_period_qa.jsonl         # Raw Q&A pairs
  war_period_qa_final.jsonl   # Deduplicated final dataset
  summary.txt                 # Human-readable summary
```

Each Q&A pair:
```json
{
  "question": "What did the Boston Evening Transcript report about the war on April 15, 1917?",
  "question_type": "explanatory",
  "ground_truths": [{"title": "Boston Evening Transcript, April 15, 1917", "ark_id": "commonwealth:abc123"}],
  "answer": "Based on the Boston Evening Transcript from April 15, 1917: ...",
  "confidence": "high",
  "period": "wwi"
}
```

## Configuration

See `war_period_config.yaml` for all options:

| Section | Key settings |
|---------|-------------|
| `war_periods` | Year ranges for WWI and WWII |
| `newspapers` | Collection URLs for Digital Commonwealth API |
| `dots_ocr` | Mode (`vllm`/`transformers`), model, prompt, vLLM URL |
| `performance` | Workers, delay, checkpoint frequency, `sample_per_year` |
| `war_keywords` | Keyword lists for article extraction scoring |
| `llm` | GPT model for Q&A generation |

## Resume Support

Every step is resumable. Checkpoints track processed record IDs (step 2), `.done` markers track downloaded issues (step 3), existing OCR files are skipped (step 4), and existing Q&A ark_ids prevent duplicates (step 6).
