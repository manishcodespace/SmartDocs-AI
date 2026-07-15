"""
services/excel_aggregation_service.py

Aggregation Engine — Deterministic, Pandas-native aggregation operations.

Responsibilities:
  - Scalar aggregations: SUM, COUNT, AVG, MIN, MAX on a filtered DataFrame
  - Group-by aggregations: per-group statistics (e.g., sum per Branch)
  - Sort & Rank: sort DataFrame by column, return top-N rows
  - Return AggregationResult for scalar ops and list[dict] for group ops

Design principles:
  - ALL arithmetic is done by Pandas — the LLM never calculates values
  - Results are deterministic and verifiable
  - Numeric conversion is attempted gracefully; NaN rows are excluded from agg
  - Group results are sorted by aggregated value (highest first for sum/avg/max/count)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from schemas.query_plan_schema import AggregationResult, AggregationSpec, GroupBySpec, SortSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalar aggregation
# ---------------------------------------------------------------------------


def aggregate(df: pd.DataFrame, spec: AggregationSpec) -> AggregationResult:
    """
    Execute a single scalar aggregation on the (filtered) DataFrame.

    Parameters
    ----------
    df   : pd.DataFrame   — the dataset (already filtered by query_service)
    spec : AggregationSpec — {type, column}

    Returns
    -------
    AggregationResult {type, column, value, label}
    """
    agg_type = spec.type.lower()

    if df.empty:
        return AggregationResult(
            type=agg_type,
            column=spec.column,
            value=0,
            label=f"{agg_type.upper()}({spec.column or '*'}) = 0 (no matching rows)",
        )

    try:
        if agg_type == "count":
            if spec.column and spec.column in df.columns:
                value: Any = int(df[spec.column].count())  # excludes NaN
            else:
                value = len(df)  # COUNT(*)
            column_label = spec.column or "*"

        else:
            if not spec.column:
                raise ValueError(
                    f"A column name is required for '{agg_type}' aggregation."
                )
            if spec.column not in df.columns:
                raise ValueError(
                    f"Column '{spec.column}' not found in the result set."
                )
            numeric = pd.to_numeric(df[spec.column], errors="coerce")
            valid_count = numeric.notna().sum()

            if valid_count == 0:
                return AggregationResult(
                    type=agg_type,
                    column=spec.column,
                    value=None,
                    label=(
                        f"{agg_type.upper()}({spec.column}) = N/A "
                        f"(column contains no numeric values)"
                    ),
                )

            if agg_type == "sum":
                value = float(numeric.sum())
            elif agg_type == "avg":
                value = float(numeric.mean())
            elif agg_type == "min":
                value = float(numeric.min())
            elif agg_type == "max":
                value = float(numeric.max())
            else:
                raise ValueError(f"Unknown aggregation type: '{agg_type}'")

            column_label = spec.column

        label = f"{agg_type.upper()}({column_label}) = {_format_value(value)}"

        logger.info("Scalar aggregation: %s", label)
        return AggregationResult(
            type=agg_type,
            column=spec.column,
            value=value,
            label=label,
        )

    except Exception as exc:
        logger.error("Aggregation failed [%s]: %s", agg_type, exc)
        raise


# ---------------------------------------------------------------------------
# Group-by aggregation
# ---------------------------------------------------------------------------


def group_and_aggregate(df: pd.DataFrame, spec: GroupBySpec) -> list[dict[str, Any]]:
    """
    Group the DataFrame by *spec.column* and compute the aggregation per group.
    Results are sorted by aggregated value (descending for sum/avg/max/count).

    Parameters
    ----------
    df   : pd.DataFrame — (filtered) dataset
    spec : GroupBySpec  — {column, aggregation: {type, column}}

    Returns
    -------
    list of dicts, e.g.:
        [
            {"Branch": "Mumbai", "sum_Outstanding Amount": 4500000.0},
            {"Branch": "Delhi",  "sum_Outstanding Amount": 3200000.0},
        ]
    """
    if df.empty:
        return []

    group_col = spec.column
    agg_type = spec.aggregation.type.lower()
    agg_col = spec.aggregation.column

    if group_col not in df.columns:
        raise ValueError(f"Group-by column '{group_col}' not found in result set.")

    # Result column label
    result_col = f"{agg_type}_{agg_col}" if agg_col else f"{agg_type}_count"

    try:
        if agg_type == "count":
            grouped = (
                df.groupby(group_col, dropna=False)
                .size()
                .reset_index(name=result_col)
            )
        else:
            if not agg_col:
                raise ValueError(f"Aggregation column required for '{agg_type}' in group-by.")
            if agg_col not in df.columns:
                raise ValueError(f"Aggregation column '{agg_col}' not found.")

            df = df.copy()
            df[agg_col] = pd.to_numeric(df[agg_col], errors="coerce")

            pandas_func = {
                "sum": "sum",
                "avg": "mean",
                "min": "min",
                "max": "max",
            }[agg_type]

            grouped = (
                df.groupby(group_col, dropna=False)[agg_col]
                .agg(pandas_func)
                .reset_index()
                .rename(columns={agg_col: result_col})
            )

        # Sort: highest first (most useful for "which branch has highest...")
        grouped = grouped.sort_values(result_col, ascending=False, na_position="last")

        # Round floats to 2 decimal places
        if grouped[result_col].dtype in (float, "float64"):
            grouped[result_col] = grouped[result_col].round(2)

        records = grouped.where(pd.notnull(grouped), other=None).to_dict(orient="records")

        logger.info(
            "Group-by complete | group='%s' | agg=%s(%s) | groups=%d",
            group_col, agg_type, agg_col or "*", len(records),
        )
        return records

    except Exception as exc:
        logger.error("Group-by aggregation failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Sort & Rank
# ---------------------------------------------------------------------------


def sort_and_rank(
    df: pd.DataFrame,
    sort_spec: SortSpec,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Sort a DataFrame by *sort_spec.column* and optionally limit rows (ranking).

    - Numeric columns are sorted numerically (not lexicographically).
    - String columns are sorted alphabetically.
    - NaN values are always placed last.

    Parameters
    ----------
    df        : pd.DataFrame — (filtered) dataset
    sort_spec : SortSpec     — {column, ascending}
    limit     : int | None   — if set, returns only the top-N rows

    Returns
    -------
    pd.DataFrame — sorted (and limited) DataFrame
    """
    if df.empty:
        return df

    col = sort_spec.column

    if col not in df.columns:
        logger.warning("Sort column '%s' not in DataFrame — skipping sort.", col)
        return df if limit is None else df.head(limit)

    try:
        # Prefer numeric sort
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > len(df) * 0.5:
            df = df.copy()
            df["__sort_key__"] = numeric
            df = df.sort_values(
                "__sort_key__",
                ascending=sort_spec.ascending,
                na_position="last",
            ).drop(columns=["__sort_key__"])
        else:
            df = df.sort_values(col, ascending=sort_spec.ascending, na_position="last")

        df = df.reset_index(drop=True)

        logger.info(
            "Sort/Rank | column='%s' | ascending=%s | limit=%s | rows_after=%d",
            col, sort_spec.ascending, limit, len(df) if limit is None else min(len(df), limit),
        )

    except Exception as exc:
        logger.warning("Sort failed on '%s': %s — returning unsorted.", col, exc)

    return df.head(limit) if limit is not None else df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Human-readable formatting for aggregated values."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
