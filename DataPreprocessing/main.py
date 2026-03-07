"""
BPL Ingestion Pipeline - Main Entry Point
BU Spark Student Team
──────────────────────────────────────────
Runs Phase 1 and/or Phase 2 based on command line arguments.

Usage:
  # Run full pipeline (Phase 1 → Phase 2)
  python main.py

  # Run only Phase 1 (fetch IDs)
  python main.py --phase 1

  # Run only Phase 2 (fetch text)
  python main.py --phase 2

  # Use a different config file
  python main.py --config my_config.yaml

  # Run with test limit (2 pages per collection)
  python main.py --test
"""

import argparse
import logging
import yaml
from pathlib import Path

import phase1_fetch_ids
import phase2_fetch_text

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BPL Newspaper Ingestion Pipeline - BU Spark"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only phase 1 or phase 2 (default: run both)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: limit to 2 pages per collection"
    )
    args = parser.parse_args()

    # Validate config exists
    if not Path(args.config).exists():
        log.error(f"Config file not found: {args.config}")
        return

    # Apply test mode override
    if args.test:
        log.info("TEST MODE: limiting to 2 pages per collection")
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["performance"]["max_pages"] = 2
        test_config = args.config.replace(".yaml", "_test.yaml")
        with open(test_config, "w") as f:
            yaml.dump(cfg, f)
        config_path = test_config
        log.info(f"Test config written to: {test_config}")
    else:
        config_path = args.config

    # Print pipeline summary
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    scope = cfg.get("scope", {})
    log.info("=" * 60)
    log.info("BPL Ingestion Pipeline - BU Spark")
    log.info(f"  Config     : {config_path}")
    log.info(f"  Date range : {scope.get('date_start')} → {scope.get('date_end')}")
    log.info(f"  Newspapers : {[n['title'] for n in cfg.get('newspapers', [])]}")
    log.info("=" * 60)

    # Run phases
    run_phase1 = args.phase in (None, 1)
    run_phase2 = args.phase in (None, 2)

    if run_phase1:
        log.info("\n>>> Starting Phase 1: Fetch Record IDs <<<\n")
        phase1_fetch_ids.run(config_path)

    if run_phase2:
        log.info("\n>>> Starting Phase 2: Fetch Text + Metadata <<<\n")
        phase2_fetch_text.run(config_path)

    log.info("\n>>> Pipeline finished <<<")

if __name__ == "__main__":
    main()
