#!/usr/bin/env python3
"""
War-Period Pseudo Ground Truth Pipeline
========================================
BU Spark Student Team — BPL RAG Evaluation Framework

Builds Q&A ground truth from BPL newspaper data for WWI and WWII.

Usage:
  python main.py                              # full pipeline
  python main.py --steps 1 2                  # specific steps
  python main.py --sample 5 --period wwi      # quick test
  python main.py --steps 3 4                  # download + OCR only
"""

import argparse
import logging
import os
import sys
import time

from pipeline import STEPS
from pipeline.shared import load_config, HAS_PIL, HAS_OPENAI, HAS_TRANSFORMERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("war_gt")


def main():
    parser = argparse.ArgumentParser(
        description="War-Period Pseudo GT Pipeline — BPL RAG Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py                                # all 7 steps, both periods
  python main.py --sample 5 --steps 1 2 5 6 7  # quick test, skip OCR
  python main.py --period wwii --steps 3 4      # download + OCR for WWII
""",
    )
    parser.add_argument("--config", default="war_period_config.yaml")
    parser.add_argument("--steps", nargs="+", type=int,
                        default=list(STEPS.keys()),
                        help="Steps to run (default: all)")
    parser.add_argument("--period", choices=["wwi", "wwii"], default=None)
    parser.add_argument("--sample", type=int, default=None,
                        help="Records per year (for testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.sample is not None:
        cfg.setdefault("performance", {})["sample_per_year"] = args.sample

    available = list(cfg["war_periods"].keys())
    if args.period:
        if args.period not in available:
            log.error(f"Unknown period '{args.period}'. Available: {available}")
            sys.exit(1)
        periods = [args.period]
    else:
        periods = available

    # Dependency summary
    dots_mode = cfg.get("dots_ocr", {}).get("mode", "vllm")
    if dots_mode == "vllm":
        dots_ok = HAS_OPENAI
        dots_str = f"vllm ({'ready' if dots_ok else 'NO -- pip install openai'})"
    else:
        dots_ok = HAS_TRANSFORMERS and HAS_PIL
        dots_str = f"transformers ({'ready' if dots_ok else 'NO -- pip install transformers torch'})"

    log.info("=" * 60)
    log.info("  War-Period Pseudo Ground Truth Pipeline")
    log.info("=" * 60)
    log.info(f"  Config   : {args.config}")
    log.info(f"  Periods  : {periods}")
    log.info(f"  Steps    : {sorted(args.steps)}")
    log.info(f"  Sample   : {args.sample or 'full dataset'}")
    log.info(f"  PIL      : {'yes' if HAS_PIL else 'NO'}")
    log.info(f"  dots.ocr : {dots_str}")
    log.info(f"  OpenAI   : {'yes' if HAS_OPENAI and os.environ.get('OPENAI_API_KEY') else 'NO -- step 6 uses templates'}")
    log.info("=" * 60)

    for step_num in sorted(args.steps):
        if step_num not in STEPS:
            log.error(f"Unknown step {step_num}. Valid: {list(STEPS.keys())}")
            continue

        name, fn = STEPS[step_num]
        log.info(f"\n>> Step {step_num}: {name}")
        t0 = time.time()
        result = fn(cfg, periods)
        log.info(f"   Step {step_num} done in {time.time() - t0:.1f}s (result: {result})\n")

    log.info("=" * 60)
    log.info("  Pipeline complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
