"""
PseudoGT Generation Pipeline — Main Entry Point
BU Spark Student Team
──────────────────────────────────────────
Generates pseudo ground truth answers for RAG evaluation queries
using the OpenAI API and the BPL cleaned newspaper dataset.

Usage:
  # Run with config file (default: config.yaml)
  python main.py

  # Override paths via CLI
  python main.py --queries test_queries.jsonl --output test_queries_gt.jsonl

  # Use a cheaper model for testing
  python main.py --model gpt-4o-mini

  # Use a different config
  python main.py --config my_config.yaml
"""

import argparse
import logging

import yaml
from pathlib import Path

from pipeline import run_generate

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="PseudoGT Generation — BPL RAG Evaluation Framework"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--queries", default=None,
        help="Override: path to input queries JSONL",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Override: path to cleaned newspaper dataset JSONL",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override: path to output JSONL file",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override: OpenAI model (e.g. gpt-4o, gpt-4o-mini)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    llm_cfg = cfg.get("llm", {})

    # CLI args override config values
    queries_path = args.queries or paths.get("queries", "ww1_ww2_test_queries.jsonl")
    dataset_path = args.dataset or paths.get("dataset", "../DataPreprocessing/bpl_clean_dataset.jsonl")
    output_path = args.output or paths.get("output", "ww1_ww2_test_queries_gt.jsonl")
    model = args.model or llm_cfg.get("model", "gpt-4o")

    # Print summary
    log.info("=" * 60)
    log.info("PseudoGT Generation — BPL RAG Evaluation Framework")
    log.info(f"  Config  : {config_path}")
    log.info(f"  Queries : {queries_path}")
    log.info(f"  Dataset : {dataset_path}")
    log.info(f"  Output  : {output_path}")
    log.info(f"  Model   : {model}")
    log.info("=" * 60)

    run_generate(queries_path, dataset_path, output_path, model=model)

    log.info("Pipeline finished")


if __name__ == "__main__":
    main()
