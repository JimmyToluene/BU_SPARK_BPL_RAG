"""OpenAI API wrapper for pseudoGT generation."""

import json
import os
import re

from openai import OpenAI

from .prompt_builder import SYSTEM_PROMPT


def create_client() -> OpenAI:
    """Create an OpenAI client using the OPENAI_API_KEY env variable.

    Raises:
        ValueError: If the API key is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def call_llm(
    client: OpenAI,
    user_prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    temperature: float = 0.2,
) -> dict:
    """Send a prompt to the OpenAI API and parse the JSON response.

    Args:
        client: OpenAI client instance.
        user_prompt: The formatted user message with query + excerpts.
        model: Model ID (e.g. gpt-4o, gpt-4o-mini).
        max_tokens: Max tokens in the response.
        temperature: Sampling temperature.

    Returns:
        Parsed dict with keys: answer, ground_truths, confidence, notes.
    """
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps its response
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "answer": raw,
            "ground_truths": [],
            "confidence": "low",
            "notes": "Failed to parse model JSON response",
        }
