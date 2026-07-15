"""
services/query_planner.py

Query Planner — Translates raw AI intent dict → typed ExecutionPlan.

Responsibilities:
  - Accept the raw dict output from excel_ai_agent.parse_question()
  - Translate every field into strongly-typed Pydantic models
  - Normalize operator names (aliases → canonical form)
  - Set sensible defaults for missing fields
  - Log the complete plan for observability

This is a pure transformation layer — no I/O, no API calls, no Pandas.
It is the single place that converts untyped AI output into a type-safe plan.
"""

from __future__ import annotations

import logging
from typing import Any

from schemas.query_plan_schema import (
    AggregationSpec,
    ExecutionPlan,
    FilterCondition,
    GroupBySpec,
    SortSpec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_INTENTS = {"filter", "aggregate", "group", "rank"}
_VALID_AGG_TYPES = {"sum", "count", "avg", "min", "max"}

# Canonical operator names — the query service uses these
_OPERATOR_ALIASES: dict[str, str] = {
    # Equality
    "eq": "==",
    "equals": "==",
    "equal": "==",
    "is": "==",
    # Inequality
    "ne": "!=",
    "not_equal": "!=",
    "not equal": "!=",
    # Numeric
    "gt": ">",
    "greater_than": ">",
    "gte": ">=",
    "greater_than_equal": ">=",
    "lt": "<",
    "less_than": "<",
    "lte": "<=",
    "less_than_equal": "<=",
    # Null checks
    "is_null": "isnull",
    "is null": "isnull",
    "isna": "isnull",
    "is_na": "isnull",
    "empty": "isnull",
    "not_null": "notnull",
    "not null": "notnull",
    "notna": "notnull",
    "not_na": "notnull",
    "not_empty": "notnull",
    # String
    "not_contains": "not_contains",
    "nin": "not_in",
    # Date aliases
    "in_month": "month",
    "for_month": "month",
    "this week": "this_week",
    "last week": "last_week",
    "this month": "this_month",
    "last month": "last_month",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_plan(ai_dict: dict[str, Any]) -> ExecutionPlan:
    """
    Translate the raw AI-produced dict into a validated ExecutionPlan.

    Parameters
    ----------
    ai_dict : dict
        Raw JSON output from excel_ai_agent.parse_question().
        Expected keys: intent, filters, aggregation, group_by, sort_by, limit, summary_template

    Returns
    -------
    ExecutionPlan — typed, normalized, ready for QueryValidator
    """
    intent = _parse_intent(ai_dict.get("intent"))
    filters = _parse_filters(ai_dict.get("filters", []))
    aggregation = _parse_aggregation(ai_dict.get("aggregation"))
    group_by = _parse_group_by(ai_dict.get("group_by"))
    sort_by = _parse_sort_by(ai_dict.get("sort_by"))
    limit = _parse_limit(ai_dict.get("limit"))
    summary_template = str(ai_dict.get("summary_template", "matching records")).strip() or "matching records"

    # Intent inference: if no explicit intent but aggregation present → "aggregate"
    if intent == "filter" and aggregation and not filters:
        intent = "aggregate"
    # If group_by present → "group"
    if group_by:
        intent = "group"
    # If sort_by + limit present and intent is still "filter" → "rank"
    if sort_by and limit and intent == "filter":
        intent = "rank"

    plan = ExecutionPlan(
        intent=intent,
        filters=filters,
        aggregation=aggregation,
        group_by=group_by,
        sort_by=sort_by,
        limit=limit,
        summary_template=summary_template,
    )

    logger.info(
        "ExecutionPlan built | intent=%s | filters=%d | agg=%s | group=%s | sort=%s | limit=%s",
        plan.intent,
        len(plan.filters),
        f"{plan.aggregation.type}({plan.aggregation.column})" if plan.aggregation else "None",
        f"{plan.group_by.column}" if plan.group_by else "None",
        f"{plan.sort_by.column} asc={plan.sort_by.ascending}" if plan.sort_by else "None",
        plan.limit,
    )

    return plan


# ---------------------------------------------------------------------------
# Private parsers
# ---------------------------------------------------------------------------


def _parse_intent(raw: Any) -> str:
    if not raw:
        return "filter"
    normalized = str(raw).strip().lower()
    return normalized if normalized in _VALID_INTENTS else "filter"


def _parse_filters(raw_filters: Any) -> list[FilterCondition]:
    if not isinstance(raw_filters, list):
        return []
    result: list[FilterCondition] = []
    for item in raw_filters:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column", "")).strip()
        if not column:
            continue
        operator_raw = str(item.get("operator", "==")).strip().lower()
        operator = _OPERATOR_ALIASES.get(operator_raw, operator_raw)
        value = item.get("value")
        result.append(FilterCondition(column=column, operator=operator, value=value))
    return result


def _parse_aggregation(raw: Any) -> AggregationSpec | None:
    if not isinstance(raw, dict):
        return None
    agg_type = str(raw.get("type", "")).strip().lower()
    if agg_type not in _VALID_AGG_TYPES:
        logger.debug("Skipping invalid aggregation type: '%s'", agg_type)
        return None
    column = raw.get("column")
    return AggregationSpec(
        type=agg_type,
        column=str(column).strip() if column else None,
    )


def _parse_group_by(raw: Any) -> GroupBySpec | None:
    if not isinstance(raw, dict):
        return None
    column = str(raw.get("column", "")).strip()
    if not column:
        return None
    agg = _parse_aggregation(raw.get("aggregation"))
    if agg is None:
        # Default: count per group
        agg = AggregationSpec(type="count", column=None)
    return GroupBySpec(column=column, aggregation=agg)


def _parse_sort_by(raw: Any) -> SortSpec | None:
    if not isinstance(raw, dict):
        return None
    column = str(raw.get("column", "")).strip()
    if not column:
        return None
    ascending_raw = raw.get("ascending", True)
    ascending = bool(ascending_raw) if not isinstance(ascending_raw, bool) else ascending_raw
    return SortSpec(column=column, ascending=ascending)


def _parse_limit(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        limit = int(raw)
        return limit if limit > 0 else None
    except (ValueError, TypeError):
        return None
