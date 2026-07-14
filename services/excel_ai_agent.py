"""
services/excel_ai_agent.py

Responsibilities:
  - Accept a user's natural-language question and the DataFrame's column names
  - Call Google Gemini to parse the question into structured filter instructions
  - Return a structured result:
      {
          "filters": [
              {"column": "Payment Status", "operator": "contains", "value": "Pending"},
              ...
          ],
          "summary_template": "customers with pending payments"
      }
  - The filters are then passed to excel_query_service.apply_filters()

This module is completely independent from the existing RAG / PDF services.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an intelligent data analyst assistant.
Your job is to convert a user's natural-language question into structured filter instructions for a Pandas DataFrame.

You will be given:
1. The column names available in the DataFrame.
2. A natural-language question from the user.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON must have exactly this shape:
{{
  "filters": [
    {{
      "column": "<exact column name from the list>",
      "operator": "<one of: ==, !=, >, >=, <, <=, contains, not_contains, startswith, endswith, in, not_in, isnull, notnull>",
      "value": "<the value to filter by, or null for isnull/notnull>"
    }}
  ],
  "summary_template": "<short human-readable description of the filter result, e.g. 'customers with pending payments for July'>"
}}

Rules:
- Only use column names from the provided list.
- If the question mentions a month (e.g. July), use 'contains' operator with the month name as the value.
- If no meaningful filter can be derived, return an empty filters array.
- Never add extra keys to the JSON.
- Never wrap the JSON in markdown code fences.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_question(
    question: str,
    column_names: list[str],
) -> dict[str, Any]:
    """
    Send *question* and *column_names* to Gemini and return structured filters.

    Parameters
    ----------
    question     : str        — the user's natural-language question
    column_names : list[str]  — headers from the uploaded DataFrame

    Returns
    -------
    dict with keys:
        filters          : list[dict]  — filter instructions for excel_query_service
        summary_template : str         — short description of the expected result
    """
    prompt = _build_prompt(question, column_names)
    raw_response = _call_gemini(prompt)
    result = _parse_json_response(raw_response)
    _validate_filters_against_columns(result.get("filters", []), column_names)
    return result


def build_summary(summary_template: str, count: int) -> str:
    """
    Combine the AI's summary template with the actual row count into a
    final human-readable sentence.
    """
    if count == 0:
        return f"No records found matching: {summary_template}."
    noun = "record" if count == 1 else "records"
    return f"Found {count} {noun} — {summary_template}."


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_prompt(question: str, column_names: list[str]) -> str:
    columns_str = "\n".join(f"  - {c}" for c in column_names)
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Available columns:\n{columns_str}\n\n"
        f"User question: {question}"
    )


def _call_gemini(prompt: str) -> str:
    """
    Invoke the Gemini model and return the raw text response.
    Uses the same API key and model settings as the rest of the project.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-google-genai is not installed. "
            "It is already a dependency of this project."
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Google Gemini API key is missing. "
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment."
        )

    chat_model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

    llm = ChatGoogleGenerativeAI(
        model=chat_model_name,
        temperature=0,          # deterministic JSON output
        max_retries=2,
    )

    response = llm.invoke(prompt)

    # Extract text from LangChain response object
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
    """
    Robustly extract a JSON object from the model's response.
    Strips markdown fences if the model incorrectly includes them.
    """
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Attempt to extract the first {...} block as a fallback
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Gemini response could not be parsed as JSON: %s", raw)
                raise ValueError(
                    "The AI returned an unexpected response format. "
                    "Please rephrase your question and try again."
                ) from exc
        else:
            logger.error("No JSON object found in Gemini response: %s", raw)
            raise ValueError(
                "The AI could not understand your question well enough to build a filter. "
                "Please be more specific."
            ) from exc

    # Ensure required keys exist with sane defaults
    if "filters" not in data:
        data["filters"] = []
    if "summary_template" not in data:
        data["summary_template"] = "matching records"

    return data


def _validate_filters_against_columns(
    filters: list[dict[str, Any]],
    column_names: list[str],
) -> None:
    """
    Log a warning for any filter that references a column not in the DataFrame.
    (The query service already handles missing columns gracefully, but we log
    it here for observability.)
    """
    lower_columns = {c.strip().lower() for c in column_names}
    for f in filters:
        col = f.get("column", "")
        if col.strip().lower() not in lower_columns:
            logger.warning(
                "AI returned filter for unknown column '%s'. "
                "Available columns: %s",
                col,
                column_names,
            )
