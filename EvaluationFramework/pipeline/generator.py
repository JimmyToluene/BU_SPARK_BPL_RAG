"""Orchestrates pseudoGT generation: load data, iterate queries, call LLM, save results."""

import logging
from datetime import datetime, timezone

from .dataset_loader import load_dataset
from .query_loader import load_queries, load_existing_output, append_result
from .retriever import find_relevant_records
from .prompt_builder import build_user_prompt
from .llm_client import create_client, call_llm

log = logging.getLogger(__name__)


def run(queries_path: str, dataset_path: str, output_path: str, model: str = "gpt-4o") -> None:
    """Run the full pseudoGT generation pipeline.

    Args:
        queries_path: Path to input queries JSONL.
        dataset_path: Path to cleaned newspaper dataset JSONL.
        output_path: Path to output JSONL with generated pseudoGT.
        model: OpenAI model ID (default: gpt-4o).
    """
    client = create_client()
    records = load_dataset(dataset_path)
    queries = load_queries(queries_path)
    existing = load_existing_output(output_path)

    for i, query_obj in enumerate(queries):
        question = query_obj["question"]
        progress = f"[{i + 1}/{len(queries)}]"

        # Skip if already processed in a previous run
        if question in existing:
            log.info(f"{progress} Skipping (already done): {question[:60]}...")
            continue

        # Pass through queries that already have ground truths
        if query_obj.get("ground_truths"):
            log.info(f"{progress} Skipping (has ground truths): {question[:60]}...")
            existing[question] = query_obj
            append_result(output_path, query_obj)
            continue

        log.info(f"{progress} Generating pseudoGT: {question[:60]}...")

        # Retrieve relevant records
        relevant = find_relevant_records(question, records)

        if not relevant:
            result = {
                "answer": "No relevant newspaper records found in the dataset for this query.",
                "ground_truths": [],
                "confidence": "low",
                "notes": "No matching records in dataset",
            }
        else:
            user_prompt = build_user_prompt(question, relevant)
            result = call_llm(client, user_prompt, model=model)

        # Build and save output record
        output_obj = {
            "question": question,
            "question_type": query_obj.get("question_type"),
            "ground_truths": result.get("ground_truths", []),
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", "low"),
            "notes": f"pseudoGT | {result.get('notes', 'generated from actual newspaper text')}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        append_result(output_path, output_obj)
        existing[question] = output_obj
        log.info(
            f"  -> confidence: {result.get('confidence', 'N/A')}, "
            f"ground_truths: {len(result.get('ground_truths', []))}"
        )

    log.info(f"Done! {len(existing)} total results written to {output_path}")
