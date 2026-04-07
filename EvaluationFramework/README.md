# BPL RAG Evaluation Framework — PseudoGT Generation

Generates **pseudo ground truth (pseudoGT)** answers and relevant document IDs for test queries, using the OpenAI API and the BPL cleaned newspaper dataset.

Part of the **BU Spark x Boston Public Library** RAG project covering WW1 (1914–1918) and WW2 (1941–1945) newspaper archives.

## Requirements

- Python 3.9+
- `pip install openai pyyaml`

## Quick Start

```bash
cd EvaluationFramework

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Edit config.yaml to set your paths and model
# Then run:
python main.py
```

## Configuration (`config.yaml`)

```yaml
paths:
  dataset:  "../DataPreprocessing/bpl_clean_dataset.jsonl"
  queries:  "ww1_ww2_test_queries.jsonl"
  output:   "ww1_ww2_test_queries_gt.jsonl"

llm:
  model:        "gpt-4o"       # gpt-4o for quality, gpt-4o-mini for cost
  max_tokens:   1000
  temperature:  0.2

retriever:
  max_records:         5       # top-K records per query
  text_preview_chars:  5000    # chars searched per record
  excerpt_chars:       2000    # chars sent to LLM per record
```

All config values can be overridden via CLI flags:

```bash
python main.py --queries test_queries.jsonl --output test_queries_gt.jsonl --model gpt-4o-mini
```

## Pipeline Components

| Module | What it does |
|--------|-------------|
| `pipeline/dataset_loader.py` | Load cleaned newspaper dataset from JSONL |
| `pipeline/query_loader.py` | Load/save test queries and output results (JSONL) |
| `pipeline/retriever.py` | Keyword-based search to find relevant records for a query |
| `pipeline/prompt_builder.py` | Build system and user prompts for the LLM |
| `pipeline/llm_client.py` | OpenAI API wrapper (create client, call model, parse JSON) |
| `pipeline/generator.py` | Orchestrate the full generation loop |
| `main.py` | CLI entry point with config loading |

## Usage Examples

```bash
# Generate pseudoGT for WW1/WW2 queries (default config)
python main.py

# Generate pseudoGT for the general test queries
python main.py --queries test_queries.jsonl --output test_queries_gt.jsonl

# Use gpt-4o-mini for cheaper testing (~10x less cost)
python main.py --model gpt-4o-mini

# Use a custom config file
python main.py --config my_config.yaml
```

## Input Format

Each line in the queries JSONL file:

```json
{
  "question": "What did Boston newspapers report about the attack on Pearl Harbor?",
  "question_type": "explanatory",
  "ground_truths": null,
  "answer": null,
  "notes": "pseudoGT_needed"
}
```

## Output Format

Each line in the output JSONL file:

```json
{
  "question": "What did Boston newspapers report about the attack on Pearl Harbor?",
  "question_type": "explanatory",
  "ground_truths": [
    {"title": "Boston Traveler, December 8 1941", "ark_id": "commonwealth:abc123"}
  ],
  "answer": "Based on the December 8 1941 edition of the Boston Traveler...",
  "confidence": "high",
  "notes": "pseudoGT | generated from actual newspaper text",
  "generated_at": "2026-04-07T00:00:00Z"
}
```

## Resume Support

The pipeline saves after **every query**. If interrupted, re-run the same command and it will skip already-processed queries automatically. Queries that already have `ground_truths` in the input are passed through unchanged.

## Cost Estimate

```
30 queries x ~1,500 input + ~300 output tokens = ~54,000 tokens total

gpt-4o      : ~$0.54
gpt-4o-mini : ~$0.05  <- recommended for testing
```

## Project Structure

```
EvaluationFramework/
  main.py                        # Entry point
  config.yaml                    # Configuration
  pipeline/
    __init__.py                  # Package exports
    dataset_loader.py            # Load newspaper dataset
    query_loader.py              # Load/save queries & results
    retriever.py                 # Keyword-based record retrieval
    prompt_builder.py            # System & user prompt construction
    llm_client.py                # OpenAI API wrapper
    generator.py                 # Generation orchestration loop
  test_queries.jsonl             # 38 general test queries
  ww1_ww2_test_queries.jsonl     # 30 WW1/WW2 queries (input)
  CLAUDE_CODE_PSEUDOGT.md        # Task specification
```
