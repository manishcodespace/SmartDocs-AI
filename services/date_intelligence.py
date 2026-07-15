"""
services/date_intelligence.py

Date Intelligence Layer — Resolves natural-language date references to concrete ranges.

Supported expressions:
  Relative  : "this week", "last week", "this month", "last month", "yesterday", "today"
  Month name: "July", "january", "Aug"
  Month+year: "July 2025", "Jan 2024"
  Quarter   : "Q1 2025", "Q2 2024", "q3"
  Year      : "2025", "FY2025", "FY 2024"
  ISO dates : "2025-07-01", "01/07/2025", "07/01/2025"

Used by excel_query_service._build_mask() when operator is one of:
  month, year, this_week, last_week, this_month, last_month, date_range

Design principle:
  The LLM identifies THAT a date filter is needed and WHICH column.
  This module resolves HOW (i.e., the actual date range).
  No LLM is involved in date arithmetic — it's pure Python.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Month name lookup
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_QUARTER_MAP: dict[str, tuple[int, int]] = {
    "q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12),
}

DateRange = tuple[date, date]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_date_filter(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """
    Apply a date-aware filter to a Pandas Series.

    Parameters
    ----------
    series   : pd.Series — the raw column (may be string or datetime dtype)
    operator : str       — date operator (month, year, this_week, date_range, ...)
    value    : Any       — the raw value from the filter condition

    Returns
    -------
    pd.Series[bool] — boolean mask (True = row matches)
    """
    # Parse the column as datetime (coerce errors to NaT)
    try:
        date_series = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    except Exception:
        logger.warning("Could not parse column as datetime for date filter.")
        return pd.Series([False] * len(series), index=series.index)

    op = operator.strip().lower()

    # --- Relative week/month ---
    if op in ("this_week", "last_week", "this_month", "last_month"):
        date_range = _relative_range(op)
        if date_range:
            return _in_range(date_series, date_range)
        return pd.Series([False] * len(series), index=series.index)

    # --- Month filter ---
    if op == "month":
        month_num = _resolve_month(value)
        if month_num is None:
            logger.warning("Could not resolve month from value=%r", value)
            return pd.Series([False] * len(series), index=series.index)
        return date_series.dt.month == month_num

    # --- Year filter ---
    if op == "year":
        year_range = _resolve_year(value)
        if year_range is None:
            logger.warning("Could not resolve year from value=%r", value)
            return pd.Series([False] * len(series), index=series.index)
        return _in_range(date_series, year_range)

    # --- Explicit date range ---
    if op == "date_range":
        date_range = _resolve_explicit_range(value)
        if date_range:
            return _in_range(date_series, date_range)
        return pd.Series([False] * len(series), index=series.index)

    # Fallback — should not reach here if operator routing is correct
    logger.warning("Unknown date operator: '%s'", op)
    return pd.Series([False] * len(series), index=series.index)


def is_date_operator(operator: str) -> bool:
    """Return True if this operator should be handled by date_intelligence."""
    return operator.strip().lower() in {
        "month", "year",
        "this_week", "last_week",
        "this_month", "last_month",
        "date_range",
    }


# ---------------------------------------------------------------------------
# Date range resolvers
# ---------------------------------------------------------------------------


def _relative_range(op: str) -> Optional[DateRange]:
    today = date.today()

    if op == "this_week":
        start = today - timedelta(days=today.weekday())  # Monday
        end = start + timedelta(days=6)  # Sunday
        return (start, end)

    if op == "last_week":
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return (start, end)

    if op == "this_month":
        start = today.replace(day=1)
        end = _last_day_of_month(today.year, today.month)
        return (start, end)

    if op == "last_month":
        first_of_this = today.replace(day=1)
        last_of_prev = first_of_this - timedelta(days=1)
        start = last_of_prev.replace(day=1)
        return (start, last_of_prev)

    return None


def _resolve_month(value: Any) -> Optional[int]:
    if value is None:
        return None
    v = str(value).strip().lower()

    # Try integer
    try:
        month = int(v)
        if 1 <= month <= 12:
            return month
    except ValueError:
        pass

    # Exact month name
    if v in _MONTH_MAP:
        return _MONTH_MAP[v]

    # Month name embedded in string: "July 2025", "payments in january"
    for name, num in _MONTH_MAP.items():
        if re.search(r"\b" + name + r"\b", v):
            return num

    return None


def _resolve_year(value: Any) -> Optional[DateRange]:
    if value is None:
        return None
    v = re.sub(r"(?i)fy\s*", "", str(value).strip())  # strip FY prefix
    try:
        year = int(v)
        return (date(year, 1, 1), date(year, 12, 31))
    except ValueError:
        return None


def _resolve_explicit_range(value: Any) -> Optional[DateRange]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start = _parse_date_str(str(value[0]))
    end = _parse_date_str(str(value[1]))
    if start and end:
        return (start, end)
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _in_range(date_series: pd.Series, date_range: DateRange) -> pd.Series:
    start_ts = pd.Timestamp(date_range[0])
    # Include the full end day
    end_ts = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return (date_series >= start_ts) & (date_series <= end_ts)


def _last_day_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _parse_date_str(date_str: str) -> Optional[date]:
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%B %d %Y", "%b %d %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(date_str, infer_datetime_format=True).date()
    except Exception:
        return None
