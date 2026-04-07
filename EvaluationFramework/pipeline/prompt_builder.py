"""System and user prompt construction for pseudoGT generation."""

SYSTEM_PROMPT = """You are a historical newspaper analyst for the Boston Public Library.
Your task is to generate pseudo ground truth (pseudoGT) answers for a RAG evaluation dataset.

You will be given:
1. A user query about WW1 or WW2 history
2. Actual newspaper excerpts from the Boston Traveler and Boston Evening Transcript

Your job is to:
1. Write a concise, accurate answer (2-4 sentences) based ONLY on the newspaper excerpts provided
2. Identify which newspaper issues are most relevant as ground truth documents
3. Return your response as JSON only — no preamble, no markdown fences

Response format (JSON only):
{
  "answer": "Your 2-4 sentence answer based on the newspaper text",
  "ground_truths": [
    {"title": "Newspaper title and date", "ark_id": "commonwealth:xxxxxxx"},
    ...
  ],
  "confidence": "high|medium|low",
  "notes": "Any caveats about OCR quality or coverage gaps"
}

Rules:
- Base your answer ONLY on the provided excerpts — do not use external knowledge
- If the excerpts do not contain relevant information, say so honestly
- Keep answers factual and cite specific dates when mentioned in the text
- confidence = high if text directly answers query, medium if partially, low if tangential"""


def build_user_prompt(
    query: str,
    records: list[dict],
    excerpt_chars: int = 2000,
) -> str:
    """Build the user message containing the query and newspaper excerpts.

    Args:
        query: The user's question.
        records: Relevant newspaper records to include as context.
        excerpt_chars: Max chars of clean_text per record.

    Returns:
        Formatted prompt string.
    """
    excerpts = []
    for rec in records:
        text_preview = rec.get("clean_text", "")[:excerpt_chars]
        excerpts.append(
            f"--- Record: {rec.get('newspaper', 'Unknown')}, "
            f"{rec.get('issue_date', 'Unknown date')} ---\n"
            f"ark_id: {rec.get('record_id', 'N/A')}\n"
            f"Text excerpt:\n{text_preview}\n"
        )

    return (
        f"Query: {query}\n\n"
        f"Here are the most relevant newspaper excerpts from the BPL archive:\n\n"
        + "\n".join(excerpts)
    )
