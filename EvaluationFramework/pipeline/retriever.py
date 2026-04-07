"""Keyword-based retrieval to find relevant newspaper records for a query."""

import re


def find_relevant_records(
    query: str,
    records: list[dict],
    max_records: int = 5,
    text_preview_chars: int = 5000,
) -> list[dict]:
    """Score and rank dataset records by keyword overlap with the query.

    Each record is scored by counting how many query keywords (length > 2)
    appear in its newspaper name, issue date, and text content.

    Args:
        query: The user's search query.
        records: Full list of newspaper records from the dataset.
        max_records: Maximum number of records to return.
        text_preview_chars: How many chars of clean_text to search per record.

    Returns:
        Top-scoring records, sorted by relevance (descending).
    """
    keywords = [w for w in re.split(r"\W+", query.lower()) if len(w) > 2]

    scored = []
    for rec in records:
        searchable = " ".join([
            rec.get("newspaper", ""),
            rec.get("issue_date", ""),
            rec.get("clean_text", "")[:text_preview_chars],
        ]).lower()

        score = sum(1 for kw in keywords if kw in searchable)
        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [rec for _, rec in scored[:max_records]]
