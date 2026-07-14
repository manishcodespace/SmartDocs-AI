"""
services/excel_query_service.py

Responsibilities:
  - Accept a Pandas DataFrame and a list of structured filter dicts
  - Apply each filter using Pandas operations
  - Return the matching rows as a list of JSON-serialisable dicts

Supported filter operators:
    ==, !=, >, >=, <, <=, contains, startswith, endswith, in, not_in, isnull, notnull

Filter dict shape (produced by excel_ai_agent.py):
    {
        "column"  : "Payment Status",   # exact column name (case-insensitive match)
        "operator": "contains",         # one of the operators above
        "value"   : "Pending"           # the comparison value (may be None for isnull/notnull)
    }

This module is completely independent from the existing RAG / PDF services.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    filters: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Apply a list of filter dicts to *df* and return (matching_rows, count).

    Parameters
    ----------
    df      : pd.DataFrame — the full dataset loaded from Excel
    filters : list[dict]   — structured filter instructions from the AI agent

    Returns
    -------
    (rows, count)
        rows  — list of dicts (each dict is one matching row)
        count — number of matching rows
    """
    if not filters:
        # No filters → return all rows
        rows = _df_to_records(df)
        return rows, len(rows)

    # Work on a copy so the cached DataFrame is never mutated
    result = df.copy()

    for f in filters:
        column_raw = f.get("column", "")
        operator = str(f.get("operator", "==")).strip().lower()
        value = f.get("value")

        # Resolve column name (case-insensitive)
        column = _resolve_column(result, column_raw)
        if column is None:
            logger.warning(
                "Filter column '%s' not found in DataFrame — skipping filter.", column_raw
            )
            continue

        try:
            mask = _build_mask(result[column], operator, value)
            result = result[mask]
        except Exception as exc:
            logger.warning(
                "Could not apply filter {column='%s', op='%s', value=%r}: %s — skipping.",
                column,
                operator,
                value,
                exc,
            )

    rows = _df_to_records(result)
    logger.info("Filters applied: %d filters → %d matching rows", len(filters), len(rows))
    return rows, len(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_column(df: pd.DataFrame, column_name: str) -> str | None:
    """
    Find the actual column name from the DataFrame headers using a
    case-insensitive comparison.  Returns None if not found.
    """
    target = column_name.strip().lower()
    for col in df.columns:
        if col.strip().lower() == target:
            return col
    return None


def _build_mask(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """
    Build a boolean mask for *series* based on *operator* and *value*.
    """
    # Normalise string values for text operations
    str_series = series.astype(str).str.strip().str.lower()

    if operator in ("==", "eq", "equals", "equal"):
        return series.astype(str).str.strip().str.lower() == str(value).strip().lower()

    elif operator in ("!=", "ne", "not_equal", "not equal"):
        return series.astype(str).str.strip().str.lower() != str(value).strip().lower()

    elif operator in (">", "gt", "greater_than"):
        return pd.to_numeric(series, errors="coerce") > float(value)

    elif operator in (">=", "gte", "greater_than_equal"):
        return pd.to_numeric(series, errors="coerce") >= float(value)

    elif operator in ("<", "lt", "less_than"):
        return pd.to_numeric(series, errors="coerce") < float(value)

    elif operator in ("<=", "lte", "less_than_equal"):
        return pd.to_numeric(series, errors="coerce") <= float(value)

    elif operator == "contains":
        return str_series.str.contains(str(value).strip().lower(), na=False, regex=False)

    elif operator == "not_contains":
        return ~str_series.str.contains(str(value).strip().lower(), na=False, regex=False)

    elif operator == "startswith":
        return str_series.str.startswith(str(value).strip().lower(), na=False)

    elif operator == "endswith":
        return str_series.str.endswith(str(value).strip().lower(), na=False)

    elif operator in ("in",):
        values_lower = [str(v).strip().lower() for v in (value if isinstance(value, list) else [value])]
        return str_series.isin(values_lower)

    elif operator in ("not_in", "nin"):
        values_lower = [str(v).strip().lower() for v in (value if isinstance(value, list) else [value])]
        return ~str_series.isin(values_lower)

    elif operator in ("isnull", "is_null", "isna", "is_na", "empty"):
        return series.isna() | (series.astype(str).str.strip() == "")

    elif operator in ("notnull", "not_null", "notna", "not_na", "not_empty"):
        return series.notna() & (series.astype(str).str.strip() != "")

    else:
        raise ValueError(f"Unsupported operator: '{operator}'")


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a DataFrame to a JSON-serialisable list of row dicts.
    NaN values are replaced with None for clean JSON output.
    """
    return df.where(pd.notnull(df), other=None).to_dict(orient="records")
