"""Step 6 — Generate Q&A pairs from extracted war articles.

Uses GPT-4o when OPENAI_API_KEY is set, otherwise falls back to templates.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .shared import log, read_jsonl, append_jsonl, HAS_OPENAI

# ── Templates (fallback when no LLM) ────────────────────────────────────────

QA_TEMPLATES = [
    {"t": "What did the {newspaper} report about the war on {date}?",
     "qt": "explanatory"},
    {"t": "Find newspaper coverage of war events from {date} in the {newspaper}.",
     "qt": "document"},
    {"t": "What war-related news appeared in the {newspaper} on {date}?",
     "qt": "explanatory"},
    {"t": "Show me {newspaper} articles from {date} about military operations.",
     "qt": "document"},
    {"t": "How did the {newspaper} cover the war effort on {date}?",
     "qt": "explanatory"},
]

# ── LLM system prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a historical newspaper analyst for the Boston Public Library.
You generate evaluation Q&A pairs from actual newspaper content.

Given a newspaper excerpt from a specific date, create ONE focused question that
the excerpt directly answers, plus a concise 2-4 sentence answer.

Rules:
- The question must be specific enough that the given newspaper excerpt is a clear ground truth
- The answer must be based ONLY on the excerpt content
- Include specific dates, names, and events mentioned in the text
- Return JSON only -- no preamble, no markdown fences

Response format:
{
  "question": "Your specific question about the war-era content",
  "question_type": "explanatory or document",
  "answer": "Your 2-4 sentence answer citing the newspaper text",
  "confidence": "high|medium|low"
}"""


def _build_gt_entry(article: dict) -> dict:
    date = article.get("issue_date", "unknown date")
    return {
        "title": f"{article['title']}, {date}",
        "ark_id": f"commonwealth:{article['ark_id']}",
    }


def _qa_from_template(article: dict, idx: int) -> dict:
    tmpl = QA_TEMPLATES[idx % len(QA_TEMPLATES)]
    newspaper = article["newspaper"]
    date = article.get("issue_date", "unknown date")
    top = article["top_segments"][0]

    return {
        "question": tmpl["t"].format(newspaper=newspaper, date=date),
        "question_type": tmpl["qt"],
        "ground_truths": [_build_gt_entry(article)],
        "answer": f"Based on the {newspaper} from {date}: {top['text'][:500]}...",
        "confidence": "medium",
        "notes": f"pseudoGT_template | war_score={top['score']} | ocr={article['ocr_source']}",
        "period": article["period"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _qa_from_llm(article: dict, client, model: str) -> dict | None:
    top = article["top_segments"][0]
    date = article.get("issue_date", "unknown date")

    prompt = (
        f"Newspaper: {article['newspaper']}\n"
        f"Date: {date}\n"
        f"ark_id: commonwealth:{article['ark_id']}\n\n"
        f"Excerpt:\n{top['text'][:2000]}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=800, temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
    except Exception as e:
        log.warning(f"    LLM failed for {article['ark_id']}: {e}")
        return None

    return {
        "question": result.get("question", ""),
        "question_type": result.get("question_type", "explanatory"),
        "ground_truths": [_build_gt_entry(article)],
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", "medium"),
        "notes": f"pseudoGT_llm | war_score={top['score']} | ocr={article['ocr_source']}",
        "period": article["period"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run(cfg: dict, periods: list[str]) -> int:
    log.info("=" * 60)
    log.info("STEP 6: Generate Q&A Pairs")
    log.info("=" * 60)

    out = cfg["output"]
    data_dir = Path(out["data_dir"])
    articles_dir = data_dir / out["articles_subdir"]
    qa_path = data_dir / out["qa_output"]

    llm_cfg = cfg.get("llm", {})
    model = llm_cfg.get("model", "gpt-4o")

    # Try to set up LLM
    use_llm = False
    client = None
    if HAS_OPENAI and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            use_llm = True
            log.info(f"  Using LLM ({model})")
        except Exception:
            pass
    if not use_llm:
        log.info("  Using template-based generation (set OPENAI_API_KEY for LLM)")

    # Existing ark_ids to avoid duplicates
    existing_arks = set()
    if qa_path.exists():
        for obj in read_jsonl(str(qa_path)):
            for gt in obj.get("ground_truths", []):
                existing_arks.add(gt.get("ark_id", ""))

    article_files = sorted(articles_dir.glob("*.jsonl"))
    article_files = [f for f in article_files if any(f.stem.startswith(p) for p in periods)]

    total = 0
    tmpl_idx = 0

    for art_file in article_files:
        articles = read_jsonl(str(art_file))
        log.info(f"  Processing {art_file.name}: {len(articles)} articles")

        for article in articles:
            ark_full = f"commonwealth:{article['ark_id']}"
            if ark_full in existing_arks or not article.get("top_segments"):
                continue

            qa = None
            if use_llm:
                qa = _qa_from_llm(article, client, model)
            if qa is None:
                qa = _qa_from_template(article, tmpl_idx)
                tmpl_idx += 1

            if qa and qa.get("question"):
                append_jsonl(str(qa_path), qa)
                existing_arks.add(ark_full)
                total += 1
                if total % 10 == 0:
                    log.info(f"    ... {total} Q&A pairs")

    log.info(f"\n  Step 6 complete: {total} new Q&A pairs")
    return total
