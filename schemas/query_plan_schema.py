"""
schemas/query_plan_schema.py

Internal Pydantic models for the enterprise Excel query pipeline.

Architecture:
  AI intent dict → QueryPlanner → ExecutionPlan
                                       ↓
                               QueryValidator → ValidationResult (resolved_plan + confidence + warnings)
                                                       ↓
                                             Pandas Execution Engine
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Filter / Aggregation / Sort specifications
# ---------------------------------------------------------------------------


class FilterCondition(BaseModel):
    """One filter condition to be applied to the DataFrame."""

    column: str = Field(..., description="Exact column name (after validation/fuzzy resolution)")
    operator: str = Field(
        ...,
        description=(
            "One of: ==, !=, >, >=, <, <=, contains, not_contains, startswith, endswith, "
            "in, not_in, isnull, notnull, between, month, year, "
            "this_week, last_week, this_month, last_month, date_range"
        ),
    )
    value: Any = Field(
        None,
        description=(
            "Filter value. May be: scalar, list (for in/between/date_range), "
            "None (for isnull/notnull/this_week etc.)"
        ),
    )


class AggregationSpec(BaseModel):
    """Specification for a scalar aggregation (SUM, COUNT, AVG, MIN, MAX)."""

    type: Literal["sum", "count", "avg", "min", "max"] = Field(..., description="Aggregation function")
    column: Optional[str] = Field(None, description="Column to aggregate. None means COUNT(*)")


class GroupBySpec(BaseModel):
    """Specification for a GROUP BY + aggregation query."""

    column: str = Field(..., description="Column to group by")
    aggregation: AggregationSpec = Field(..., description="Aggregation to compute per group")


class SortSpec(BaseModel):
    """Sort order specification."""

    column: str = Field(..., description="Column to sort by")
    ascending: bool = Field(True, description="True = ascending, False = descending")


# ---------------------------------------------------------------------------
# Execution Plan
# ---------------------------------------------------------------------------


class ExecutionPlan(BaseModel):
    """
    The complete, typed, validated query plan produced by QueryPlanner
    and refined by QueryValidator.

    This is the single source of truth passed to the Pandas execution engine.
    """

    intent: Literal["filter", "aggregate", "group", "rank"] = Field(
        "filter",
        description=(
            "filter   = return rows matching conditions\n"
            "aggregate = return a single computed value\n"
            "group    = return per-group statistics\n"
            "rank     = return rows sorted by a value (top-N)"
        ),
    )
    filters: list[FilterCondition] = Field(default_factory=list)
    aggregation: Optional[AggregationSpec] = None
    group_by: Optional[GroupBySpec] = None
    sort_by: Optional[SortSpec] = None
    limit: Optional[int] = Field(None, ge=1, description="Max rows to return")
    summary_template: str = Field("matching records", description="Short human-readable description")


# ---------------------------------------------------------------------------
# Schema (per-column metadata)
# ---------------------------------------------------------------------------


class ColumnSchema(BaseModel):
    """Metadata about a single DataFrame column."""

    name: str
    dtype: Literal["numeric", "date", "string", "boolean"] = "string"
    sample_values: list[Any] = Field(default_factory=list)
    nullable: bool = False
    min_value: Any = None
    max_value: Any = None
    unique_count: Optional[int] = None


class DatasetSchema(BaseModel):
    """Full schema for an uploaded Excel file (one sheet)."""

    file_id: str
    sheet_name: str
    total_rows: int
    columns: list[ColumnSchema] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Output of QueryValidator.validate()."""

    is_valid: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0 = no confidence, 1.0 = certain")
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    resolved_plan: ExecutionPlan = Field(..., description="Plan with fuzzy-resolved column names")


# ---------------------------------------------------------------------------
# Aggregation result (runtime, not a plan spec)
# ---------------------------------------------------------------------------


class AggregationResult(BaseModel):
    """Output of excel_aggregation_service.aggregate()."""

    type: str = Field(..., description="Aggregation type (sum/count/avg/min/max)")
    column: Optional[str] = Field(None, description="Column aggregated (None for count(*))")
    value: Any = Field(..., description="The computed scalar value")
    label: str = Field(..., description="Human-readable label, e.g. 'SUM(Outstanding Amount) = 4,500,000'")
