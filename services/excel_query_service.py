"""
services/excel_query_service.py  (upgraded)

Responsibilities:
  - Accept a Pandas DataFrame and a list of structured filter dicts
  - Apply each filter using deterministic Pandas operations
  - Route date operators to date_intelligence for precise date arithmetic
  - Return matching rows as a JSON-serialisable list of dicts

Supported operators:
  Equality  : ==, !=
  Numeric   : >, >=, <, <=, between
  String    : contains, not_contains, startswith, endswith
  Set       : in, not_in
  Null      : isnull, notnull
  Date      : month, year, this_week, last_week, this_month, last_month, date_range

Changes vs original:
  - Date operators now routed to date_intelligence.apply_date_filter()
  - 'between' operator added for numeric ranges
  - _resolve_column() kept (exact case-insensitive) — fuzzy resolution is in QueryValidator
  - Logging includes operator + value for observability

Filter dict shape (produced by query_planner.py after validation):
  {
      "column"  : "Payment Status",
      "operator": "contains",
      "value"   : "Pending"
  }
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from services.date_intelligence import apply_date_filter, is_date_operator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    filters: list[dict[str, Any]],
) -> tuple[pd.DataFrame, int]:
    """
    Apply a list of filter dicts to *df* and return (result_df, count).

    Parameters
    ----------
    df      : pd.DataFrame — the full (or pre-filtered) dataset
    filters : list[dict]   — structured filter instructions

    Returns
    -------
    (result_df, count)
        result_df — filtered DataFrame (preserves columns + dtypes)
        count     — number of matching rows
    """
    if not filters:
        return df.copy(), len(df)

    result = df.copy()

    for f in filters:
        column_raw = f.get("column", "")
        operator = str(f.get("operator", "==")).strip().lower()
        value = f.get("value")

        column = _resolve_column(result, column_raw)
        if column is None:
            logger.warning(
                "Filter column '%s' not found in DataFrame (after validation) — skipping.",
                column_raw,
            )
            continue

        try:
            mask = _build_mask(result[column], operator, value)
            result = result[mask]
            logger.debug(
                "Filter applied: %s %s %r → %d rows remain",
                column, operator, value, len(result),
            )
        except Exception as exc:
            logger.warning(
                "Could not apply filter {column='%s', op='%s', value=%r}: %s — skipping.",
                column, operator, value, exc,
            )

    count = len(result)
    logger.info(
        "Filters complete: %d filter(s) applied → %d matching row(s)",
        len(filters), count,
    )
    return result, count


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a DataFrame to a JSON-serialisable list of row dicts.
    NaN values are replaced with None for clean JSON output.
    """
    return df.where(pd.notnull(df), other=None).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_column(df: pd.DataFrame, column_name: str) -> str | None:
    """
    Find the actual column using exact case-insensitive comparison.
    Fuzzy resolution is handled upstream in QueryValidator.
    Returns None if not found.
    """
    target = column_name.strip().lower()
    for col in df.columns:
        if col.strip().lower() == target:
            return col
    return None


def _build_mask(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """
    Build a boolean mask for *series* based on *operator* and *value*.

    Date operators are routed to date_intelligence.apply_date_filter().
    All other operators are handled inline with Pandas.
    """
    # --- Date operators ---
    if is_date_operator(operator):
        return apply_date_filter(series, operator, value)

    # --- String normalisation for text operators ---
    str_series = series.astype(str).str.strip().str.lower()

    # --- Equality ---
    if operator in ("==", "eq", "equals", "equal", "is"):
        return series.astype(str).str.strip().str.lower() == str(value).strip().lower()

    if operator in ("!=", "ne", "not_equal", "not equal"):
        return series.astype(str).str.strip().str.lower() != str(value).strip().lower()

    # --- Numeric comparisons ---
    if operator in (">", "gt", "greater_than"):
        return pd.to_numeric(series, errors="coerce") > float(value)

    if operator in (">=", "gte", "greater_than_equal"):
        return pd.to_numeric(series, errors="coerce") >= float(value)

    if operator in ("<", "lt", "less_than"):
        return pd.to_numeric(series, errors="coerce") < float(value)

    if operator in ("<=", "lte", "less_than_equal"):
        return pd.to_numeric(series, errors="coerce") <= float(value)

    if operator == "between":
        if isinstance(value, (list, tuple)) and len(value) == 2:
            numeric = pd.to_numeric(series, errors="coerce")
            return (numeric >= float(value[0])) & (numeric <= float(value[1]))
        raise ValueError(
            f"Operator 'between' requires a list of two values [low, high], got: {value!r}"
        )

    # --- String operators ---
    if operator == "contains":
        return str_series.str.contains(str(value).strip().lower(), na=False, regex=False)

    if operator == "not_contains":
        return ~str_series.str.contains(str(value).strip().lower(), na=False, regex=False)

    if operator == "startswith":
        return str_series.str.startswith(str(value).strip().lower(), na=False)

    if operator == "endswith":
        return str_series.str.endswith(str(value).strip().lower(), na=False)

    # --- Set operators ---
    if operator == "in":
        values_lower = [str(v).strip().lower() for v in (value if isinstance(value, list) else [value])]
        return str_series.isin(values_lower)

    if operator in ("not_in", "nin"):
        values_lower = [str(v).strip().lower() for v in (value if isinstance(value, list) else [value])]
        return ~str_series.isin(values_lower)

    # --- Null operators ---
    if operator in ("isnull", "is_null", "isna", "is_na", "empty"):
        return series.isna() | (series.astype(str).str.strip() == "")

    if operator in ("notnull", "not_null", "notna", "not_na", "not_empty"):
        return series.notna() & (series.astype(str).str.strip() != "")

    raise ValueError(f"Unsupported operator: '{operator}'")
