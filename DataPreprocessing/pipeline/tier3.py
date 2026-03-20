"""
Tier 3 — AI-based OCR correction using Llama-2-13B-OCR
=======================================================
Fine-tuned on BLN600 (19th-century English newspapers) — the same
dataset used to build the Tier 2 wordlist.

Install dependencies:
    pip install transformers torch accelerate

If you have a GPU with 16GB+ VRAM, the model loads in float16 automatically.
On CPU it loads in float32 and will be slower but still functional.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID   = "pykale/Llama-2-13B-OCR"
CHUNK_SIZE = 1500

PROMPT_TEMPLATE = """### Instruction:
Correct the OCR errors in the following 19th-century historical text.
Preserve archaic spelling (e.g. publick, hath, ye, honour, connexion).
Do not modernise vocabulary or grammar.
If a passage cannot be confidently reconstructed, leave it as-is.
Return only the corrected text with no commentary.

### Input:
{text}

### Response:
"""


def load_model(model_id: str = MODEL_ID, verbose: bool = True):
    """
    Load the Llama-2-13B-OCR model and tokenizer.
    Auto-detects GPU (float16) or CPU (float32).
    Call once at startup and pass the returned objects to tier3_clean().

    Returns:
        (model, tokenizer, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    if verbose:
        print(f"[Tier 3] Loading model on {device.upper()} ({dtype})...")
        if device == "cpu":
            print("  Note: CPU inference is slow for a 13B model. "
                  "Consider running on a GPU machine for large corpora.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",   # distributes across all available GPUs/CPU automatically
    )
    model.eval()

    if verbose:
        print(f"[Tier 3] Model loaded.")

    return model, tokenizer, device


def _split_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into fixed-size chunks at word boundaries."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        boundary = text.rfind(" ", start, end)
        if boundary == -1:
            boundary = end

        chunks.append(text[start:boundary])
        start = boundary + 1

    return chunks


def tier3_clean(
    text: str,
    model,
    tokenizer,
    device: str,
    chunk_size: int = CHUNK_SIZE,
    max_new_tokens: int = 1024,
    verbose: bool = True,
) -> str:
    """
    Tier 3 cleaning — Llama-2-13B-OCR contextual correction.
    Assumes text has already been through Tier 1 and Tier 2.

    Args:
        text:           Tier 2 cleaned text.
        model:          Loaded model from load_model().
        tokenizer:      Loaded tokenizer from load_model().
        device:         Device string from load_model().
        chunk_size:     Max characters per inference call (default 1500).
        max_new_tokens: Max tokens the model can generate per chunk.
        verbose:        Print progress per chunk.

    Returns:
        Corrected text.
    """
    chunks = _split_chunks(text, chunk_size)

    if verbose:
        print(f"[Tier 3] {len(chunks)} chunk(s) to process...")

    corrected_chunks = []

    for i, chunk in enumerate(chunks, 1):
        if verbose:
            print(f"  Chunk {i}/{len(chunks)} ({len(chunk)} chars)...")

        prompt = PROMPT_TEMPLATE.format(text=chunk)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # deterministic output
                temperature=1.0,
                repetition_penalty=1.1,   # reduces repetitive OCR artefacts
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens, not the prompt
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        corrected  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        corrected_chunks.append(corrected)

    return " ".join(corrected_chunks)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(
        description=(
            "Tier 3 cleaning (Llama-2-13B-OCR) on a JSON file.\n"
            "Adds a new field (default: tier_3_cleaned) and writes back in-place "
            "unless --out is provided.\n\n"
            "Tier 3 assumes input text has already been through Tier 1 and Tier 2."
        )
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to input JSON file (e.g. .../1900.json).",
    )
    parser.add_argument(
        "--field",
        default="tier_2_cleaned",
        help="Record field to clean (default: tier_2_cleaned).",
    )
    parser.add_argument(
        "--new_field",
        default="tier_3_cleaned",
        help="Field name to store Tier 3 output (default: tier_3_cleaned).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output file path. If omitted, modifies input file in-place.",
    )
    parser.add_argument(
        "--overwrite_input_field",
        action="store_true",
        help="Also overwrite the input --field with the Tier 3 cleaned text.",
    )
    parser.add_argument(
        "--model_id",
        default=MODEL_ID,
        help=f"HuggingFace model id (default: {MODEL_ID}).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in characters (default: {CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens to generate per chunk (default: 1024).",
    )
    parser.add_argument(
        "--no_verbose",
        action="store_true",
        help="Reduce console output during processing.",
    )
    parser.add_argument(
        "--limit_records",
        type=int,
        default=None,
        help="Optional: limit number of records for a test run.",
    )
    args = parser.parse_args()

    in_path = _Path(args.path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    out_path = _Path(args.out) if args.out else in_path

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", [])
    else:
        # If the JSON is just a list of records
        records = data
        data = {"records": records}

    if not isinstance(records, list):
        raise SystemExit("Expected records to be a list (or top-level list).")

    if args.limit_records is not None:
        records = records[: args.limit_records]

    verbose = not args.no_verbose

    # Load model once
    model, tokenizer, device = load_model(model_id=args.model_id, verbose=verbose)

    cleaned_count = 0
    missing_count = 0

    for rec in records:
        if not isinstance(rec, dict):
            continue
        text = rec.get(args.field, "")
        if not text:
            missing_count += 1
            continue

        cleaned = tier3_clean(
            text=text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            chunk_size=args.chunk_size,
            max_new_tokens=args.max_new_tokens,
            verbose=verbose,
        )
        rec[args.new_field] = cleaned

        if args.overwrite_input_field:
            rec[args.field] = cleaned

        cleaned_count += 1

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Tier 3 cleaned {cleaned_count} record(s).")
    if missing_count:
        print(f"Skipped {missing_count} record(s) due to missing/empty '{args.field}'.")
    print(f"Saved: {out_path}")