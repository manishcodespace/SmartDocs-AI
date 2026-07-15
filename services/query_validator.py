"""
services/query_validator.py

Query Validator — Validates an ExecutionPlan against the DatasetSchema.

Responsibilities:
  1. Column resolution: exact match → fuzzy match (difflib) → warn + skip
  2. Operator / type compatibility checks (e.g., > on a string column)
  3. Aggregation column validation
  4. Confidence scoring: 0.0 (no confidence) → 1.0 (certain)
  5. Human-readable warnings for the API response
  6. Returns a resolved plan with corrected column names

This is a pure business logic layer — no I/O, no API calls, no Pandas.
"""

from __future__ import annotations

import difflib
import logging
from copy import deepcopy
from typing import Optional

from schemas.query_plan_schema import (
    AggregationSpec,
    ColumnSchema,
    DatasetSchema,
    ExecutionPlan,
    FilterCondition,
    GroupBySpec,
    SortSpec,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type/operator compatibility tables
# ---------------------------------------------------------------------------

# Operators that only make sense on numeric columns
_NUMERIC_ONLY_OPS = {">", ">=", "<", "<=", "between"}

# Operators that only make sense on date columns
_DATE_ONLY_OPS = {
    "date_range",
    "month",
    "year",
    "this_week",
    "last_week",
    "this_month",
    "last_month",
}

# Aggregation types that require a numeric column
_NUMERIC_AGGS = {"sum", "avg"}

# Fuzzy match cutoff (0.0–1.0): lower = more permissive
_FUZZY_CUTOFF = 0.6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(plan: ExecutionPlan, schema: DatasetSchema) -> ValidationResult:
    """
    Validate *plan* against *schema*.

    Returns
    -------
    ValidationResult with:
      - is_valid       : bool — True if plan can be executed (errors == [])
      - confidence     : float — 0.0–1.0 aggregate confidence score
      - warnings       : list[str] — non-fatal issues for the user
      - errors         : list[str] — fatal issues (filter/agg skipped)
      - resolved_plan  : ExecutionPlan — plan with fuzzy-resolved column names
    """
    warnings: list[str] = []
    errors: list[str] = []
    confidence_factors: list[float] = []

    resolved_plan = deepcopy(plan)

    col_names = [c.name for c in schema.columns]
    col_map: dict[str, ColumnSchema] = {c.name.strip().lower(): c for c in schema.columns}

    # ------------------------------------------------------------------
    # 1. Validate filters
    # ------------------------------------------------------------------
    resolved_filters: list[FilterCondition] = []

    for f in resolved_plan.filters:
        resolved_col, col_score, col_warn = _resolve_column(f.column, col_names, col_map)
        confidence_factors.append(col_score)

        if col_warn:
            warnings.append(col_warn)

        if resolved_col is None:
            errors.append(
                f"Filter column '{f.column}' not found in the dataset. "
                f"Available columns: {_format_columns(col_names)}."
            )
            continue

        col_schema = col_map[resolved_col.strip().lower()]

        # Operator/type compatibility
        op_warn, op_score = _check_op_compat(f.operator, col_schema)
        if op_warn:
            warnings.append(op_warn)
        confidence_factors.append(op_score)

        resolved_filters.append(
            FilterCondition(column=resolved_col, operator=f.operator, value=f.value)
        )

    resolved_plan.filters = resolved_filters

    # ------------------------------------------------------------------
    # 2. Validate aggregation
    # ------------------------------------------------------------------
    if resolved_plan.aggregation:
        agg = resolved_plan.aggregation
        if agg.column:
            resolved_col, col_score, col_warn = _resolve_column(agg.column, col_names, col_map)
            confidence_factors.append(col_score)

            if col_warn:
                warnings.append(col_warn)

            if resolved_col:
                col_schema = col_map[resolved_col.strip().lower()]
                if agg.type in _NUMERIC_AGGS and col_schema.dtype != "numeric":
                    warnings.append(
                        f"Aggregation '{agg.type.upper()}' on column '{resolved_col}' "
                        f"(inferred type: {col_schema.dtype}) — expected a numeric column. "
                        f"Pandas will attempt numeric conversion; non-numeric values will be ignored."
                    )
                    confidence_factors.append(0.55)
                resolved_plan.aggregation = AggregationSpec(type=agg.type, column=resolved_col)
            else:
                errors.append(
                    f"Aggregation column '{agg.column}' not found in the dataset."
                )
                resolved_plan.aggregation = None
        else:
            # COUNT(*) — no column needed
            confidence_factors.append(1.0)

    # ------------------------------------------------------------------
    # 3. Validate group_by
    # ------------------------------------------------------------------
    if resolved_plan.group_by:
        gb = resolved_plan.group_by

        # Group column
        resolved_gcol, gcol_score, gcol_warn = _resolve_column(gb.column, col_names, col_map)
        confidence_factors.append(gcol_score)
        if gcol_warn:
            warnings.append(gcol_warn)

        if resolved_gcol is None:
            errors.append(f"Group-by column '{gb.column}' not found.")
            resolved_plan.group_by = None
        else:
            # Aggregation column inside group_by
            gb_agg = gb.aggregation
            if gb_agg.column:
                resolved_acol, acol_score, acol_warn = _resolve_column(
                    gb_agg.column, col_names, col_map
                )
                confidence_factors.append(acol_score)
                if acol_warn:
                    warnings.append(acol_warn)

                if resolved_acol:
                    col_schema = col_map[resolved_acol.strip().lower()]
                    if gb_agg.type in _NUMERIC_AGGS and col_schema.dtype != "numeric":
                        warnings.append(
                            f"Group aggregation '{gb_agg.type.upper()}' on '{resolved_acol}' "
                            f"(type: {col_schema.dtype}) requires a numeric column."
                        )
                        confidence_factors.append(0.55)
                    resolved_plan.group_by = GroupBySpec(
                        column=resolved_gcol,
                        aggregation=AggregationSpec(type=gb_agg.type, column=resolved_acol),
                    )
                else:
                    errors.append(
                        f"Group aggregation column '{gb_agg.column}' not found."
                    )
                    resolved_plan.group_by = None
            else:
                # COUNT per group — no agg column needed
                resolved_plan.group_by = GroupBySpec(column=resolved_gcol, aggregation=gb_agg)

    # ------------------------------------------------------------------
    # 4. Validate sort_by
    # ------------------------------------------------------------------
    if resolved_plan.sort_by:
        sb = resolved_plan.sort_by
        resolved_scol, scol_score, scol_warn = _resolve_column(sb.column, col_names, col_map)
        confidence_factors.append(scol_score)

        if scol_warn:
            warnings.append(scol_warn)

        if resolved_scol:
            resolved_plan.sort_by = SortSpec(column=resolved_scol, ascending=sb.ascending)
        else:
            warnings.append(
                f"Sort column '{sb.column}' not found — sorting skipped. "
                f"Results will be returned in original order."
            )
            resolved_plan.sort_by = None

    # ------------------------------------------------------------------
    # 5. Empty plan warning
    # ------------------------------------------------------------------
    has_filter = bool(resolved_plan.filters)
    has_agg = resolved_plan.aggregation is not None
    has_group = resolved_plan.group_by is not None

    if not has_filter and not has_agg and not has_group:
        warnings.append(
            "No filters, aggregation, or grouping could be extracted from your question. "
            "All rows will be returned. Try rephrasing with more specific conditions."
        )
        confidence_factors.append(0.25)

    # ------------------------------------------------------------------
    # 6. Compute overall confidence
    # ------------------------------------------------------------------
    if confidence_factors:
        confidence = round(sum(confidence_factors) / len(confidence_factors), 3)
    else:
        confidence = 0.5

    confidence = max(0.0, min(1.0, confidence))

    is_valid = len(errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        confidence=confidence,
        warnings=warnings,
        errors=errors,
        resolved_plan=resolved_plan,
    )

    logger.info(
        "Validation complete | valid=%s | confidence=%.3f | warnings=%d | errors=%d",
        is_valid,
        confidence,
        len(warnings),
        len(errors),
    )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_column(
    column_name: str,
    col_names: list[str],
    col_map: dict[str, ColumnSchema],
) -> tuple[Optional[str], float, Optional[str]]:
    """
    Resolve *column_name* to an actual DataFrame column.

    Returns
    -------
    (resolved_name, confidence_score, warning_message)
      - resolved_name : str | None  — the actual column, or None if not found
      - confidence    : float       — 1.0 exact, 0.7 fuzzy, 0.0 not found
      - warning       : str | None  — warning message for fuzzy/missing
    """
    if not column_name or not column_name.strip():
        return None, 0.0, None

    target = column_name.strip().lower()

    # Exact match (case-insensitive)
    for col in col_names:
        if col.strip().lower() == target:
            return col, 1.0, None

    # Fuzzy match via difflib
    candidates = [c.strip().lower() for c in col_names]
    close_matches = difflib.get_close_matches(target, candidates, n=1, cutoff=_FUZZY_CUTOFF)

    if close_matches:
        matched_lower = close_matches[0]
        for col in col_names:
            if col.strip().lower() == matched_lower:
                warning = (
                    f"Column '{column_name}' was not found exactly. "
                    f"Using '{col}' as the closest match (fuzzy resolution, ~70% confidence). "
                    f"Rename your question to use the exact column name for best accuracy."
                )
                return col, 0.7, warning

    return None, 0.0, None


def _check_op_compat(
    operator: str,
    col_schema: ColumnSchema,
) -> tuple[Optional[str], float]:
    """
    Check if *operator* is compatible with the column's inferred type.

    Returns
    -------
    (warning_message, confidence_factor)
    """
    op = operator.strip().lower()

    if op in _NUMERIC_ONLY_OPS and col_schema.dtype not in ("numeric",):
        warning = (
            f"Operator '{op}' is designed for numeric columns, "
            f"but '{col_schema.name}' has type '{col_schema.dtype}'. "
            f"Pandas will attempt numeric conversion — non-numeric values will be treated as NaN."
        )
        return warning, 0.6

    if op in _DATE_ONLY_OPS and col_schema.dtype not in ("date",):
        warning = (
            f"Operator '{op}' is date-based, "
            f"but '{col_schema.name}' has type '{col_schema.dtype}'. "
            f"Date parsing will be attempted automatically."
        )
        return warning, 0.65

    return None, 1.0


def _format_columns(col_names: list[str]) -> str:
    if len(col_names) <= 6:
        return ", ".join(f"'{c}'" for c in col_names)
    shown = ", ".join(f"'{c}'" for c in col_names[:6])
    return f"{shown}, ... (+{len(col_names) - 6} more)"
