"""
services/excel_ai_agent.py  (upgraded)

Responsibilities:
  - Accept a user's natural-language question and the DatasetSchema
  - Call Google Gemini (temperature=0) to parse the question into structured intent JSON
  - Return a rich intent dict covering:
      intent, filters, aggregation, group_by, sort_by, limit, summary_template
  - Build human-readable summaries using real aggregated values (no hallucination)

Changes vs original:
  - Prompt now includes column types, sample values, and domain context
  - JSON schema extended with intent / aggregation / group_by / sort_by / limit
  - build_rich_summary() replaces build_summary() — uses actual computed values
  - Schema-aware prompt construction via DatasetSchema

This module is completely independent from the existing RAG / PDF services.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an intelligent data analyst for a loan collection management system.
Your job is to convert the user's natural-language question into a structured JSON query plan.

You will be given:
1. The dataset's column names with their inferred types and sample values.
2. A natural-language question from the user.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON must follow this exact schema:
{{
  "intent": "<filter | aggregate | group | rank>",
  "filters": [
    {{
      "column": "<exact column name from the list below>",
      "operator": "<one of: ==, !=, >, >=, <, <=, contains, not_contains, startswith, endswith, in, not_in, isnull, notnull, between, month, year, this_week, last_week, this_month, last_month, date_range>",
      "value": "<scalar value, list for in/between/date_range, null for isnull/notnull/this_week/etc.>"
    }}
  ],
  "aggregation": {{
    "type": "<sum | count | avg | min | max>",
    "column": "<column name, or null for count(*)>"
  }},
  "group_by": {{
    "column": "<column to group by>",
    "aggregation": {{
      "type": "<sum | count | avg | min | max>",
      "column": "<aggregation column>"
    }}
  }},
  "sort_by": {{
    "column": "<column to sort by>",
    "ascending": <true | false>
  }},
  "limit": <integer or null>,
  "summary_template": "<short human-readable description of the query result>"
}}

Intent selection rules:
  filter    → return rows matching conditions (most questions)
  aggregate → return a single computed value (SUM, COUNT, AVG, total, average, how many)
  group     → return per-group statistics (by branch, by region, per category)
  rank      → return rows sorted by a value (top N, highest, lowest, most, least)

Operator selection rules:
  ==        → exact string/number match
  contains  → partial text match (preferred for status fields like "Pending", "Paid")
  >  >=     → numeric greater than (for DPD, amount, score)
  <  <=     → numeric less than
  month     → filter by month name (value = "July", "January", etc.)
  year      → filter by year (value = 2025)
  this_week / last_week / this_month / last_month → relative date windows (value = null)
  date_range → explicit range (value = ["2025-07-01", "2025-07-31"])
  in        → value is a list (e.g., ["Pending", "Overdue"])
  isnull    → field is empty/null (value = null)
  notnull   → field has a value (value = null)

Critical rules:
  - ONLY use column names from the provided list — never invent column names.
  - "aggregation", "group_by", "sort_by", "limit" should be null/omitted when not needed.
  - "filters" should be an empty list [] when not needed.
  - For "total outstanding amount" → intent=aggregate, aggregation={{type:"sum", column:"<outstanding column>"}}.
  - For "which branch has highest overdue" → intent=group, group_by={{column:"Branch", aggregation:{{type:"sum", column:"<overdue column>"}}}}.
  - For "top 5 customers by DPD" → intent=rank, sort_by={{column:"DPD", ascending:false}}, limit:5.
  - For "customers with pending payment in July" → intent=filter with two filters: one for payment status, one for date.
  - Never add extra JSON keys.
  - Never wrap in markdown code fences.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_question(
    question: str,
    schema: Any,  # DatasetSchema — using Any to avoid circular import
) -> dict[str, Any]:
    """
    Parse a natural-language question into structured intent JSON.

    Parameters
    ----------
    question : str          — the user's question
    schema   : DatasetSchema — column names, types, sample values

    Returns
    -------
    dict with keys: intent, filters, aggregation, group_by, sort_by, limit, summary_template
    """
    prompt = _build_prompt(question, schema)
    raw_response = _call_gemini(prompt)
    result = _parse_json_response(raw_response)
    _log_column_warnings(result.get("filters", []), schema)
    return result


def build_rich_summary(
    template: str,
    count: int,
    agg_result: Any = None,   # AggregationResult | None
    group_result: Any = None, # list[dict] | None
    intent: str = "filter",
) -> str:
    """
    Build a human-readable summary using REAL computed values — no hallucination.

    Parameters
    ----------
    template     : str            — base description from AI
    count        : int            — number of matching rows
    agg_result   : AggregationResult | None
    group_result : list[dict] | None
    intent       : str

    Returns
    -------
    str — rich, accurate summary
    """
    if intent == "aggregate" and agg_result:
        return f"Result: {agg_result.label}."

    if intent == "group" and group_result:
        if group_result:
            top = group_result[0]
            keys = list(top.keys())
            group_col = keys[0] if keys else "Group"
            value_col = keys[1] if len(keys) > 1 else None
            top_name = top.get(group_col, "Unknown")
            top_value = top.get(value_col, "N/A") if value_col else "N/A"
            return (
                f"Group-by result — {len(group_result)} group(s). "
                f"Highest: {group_col} '{top_name}' with {value_col}={top_value}. "
                f"Full breakdown included in the table below."
            )
        return "No groups found."

    if intent == "rank":
        noun = "record" if count == 1 else "records"
        return f"Top {count} {noun} ranked by {template}."

    if count == 0:
        return f"No records found matching: {template}."

    noun = "record" if count == 1 else "records"
    suffix = ""
    if agg_result:
        suffix = f" {agg_result.label}."
    return f"Found {count} {noun} — {template}.{suffix}"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_prompt(question: str, schema: Any) -> str:
    """Build the full prompt with schema context."""
    columns_block = _format_schema(schema)
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Dataset columns:\n{columns_block}\n\n"
        f"User question: {question}"
    )


def _format_schema(schema: Any) -> str:
    """
    Format DatasetSchema into a readable prompt block.

    Example output:
      - Customer Name [string] (e.g., "Rahul Sharma", "Priya Patel")
      - DPD [numeric] (range: 0 – 365, e.g., 45, 90, 120)
      - Payment Status [string] (e.g., "Pending", "Paid", "Overdue")
      - Payment Date [date] (range: 2025-01-01 – 2025-07-15)
    """
    if hasattr(schema, "columns"):
        lines = []
        for col in schema.columns:
            samples = col.sample_values[:4]
            sample_str = ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in samples)
            range_info = ""
            if col.min_value is not None and col.max_value is not None:
                range_info = f" (range: {col.min_value} – {col.max_value})"
            sample_info = f' (e.g., {sample_str})' if sample_str else ""
            lines.append(f"  - {col.name} [{col.dtype}]{range_info}{sample_info}")
        return "\n".join(lines)
    # Fallback: schema is just a list of column names
    if isinstance(schema, list):
        return "\n".join(f"  - {c}" for c in schema)
    return str(schema)


def _call_gemini(prompt: str) -> str:
    """Invoke Gemini with temperature=0 and return raw text."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-google-genai is not installed. It is a dependency of this project."
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Google Gemini API key is missing. "
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment."
        )

    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,        # deterministic — critical for structured output
        max_retries=2,
    )

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list):
        parts = []
        for part in response.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(response.content)


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Robustly extract and parse JSON from the model's response."""
    text = raw.strip()

    # Strip markdown fences if model incorrectly includes them
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Fallback: extract first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Gemini response could not be parsed as JSON: %s", raw[:200])
                raise ValueError(
                    "The AI returned an unexpected response format. "
                    "Please rephrase your question and try again."
                ) from exc
        else:
            logger.error("No JSON object found in Gemini response: %s", raw[:200])
            raise ValueError(
                "The AI could not understand your question. "
                "Please rephrase it with more specific terms."
            ) from exc

    # Ensure required keys with sane defaults
    data.setdefault("intent", "filter")
    data.setdefault("filters", [])
    data.setdefault("aggregation", None)
    data.setdefault("group_by", None)
    data.setdefault("sort_by", None)
    data.setdefault("limit", None)
    data.setdefault("summary_template", "matching records")

    return data


def _log_column_warnings(
    filters: list[dict[str, Any]],
    schema: Any,
) -> None:
    """Log warnings for filter columns that may not match the schema."""
    if not hasattr(schema, "columns"):
        return
    available = {c.name.strip().lower() for c in schema.columns}
    for f in filters:
        col = f.get("column", "")
        if col.strip().lower() not in available:
            logger.warning(
                "AI returned filter for unrecognized column '%s' (will be resolved by QueryValidator). "
                "Available: %s",
                col,
                [c.name for c in schema.columns][:8],
            )
