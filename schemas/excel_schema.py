"""
schemas/excel_schema.py

Pydantic models for the Excel AI Query module.
Completely independent from the existing PDF/RAG schemas.

Changes vs original:
  - ExcelQueryResponse extended with:
      aggregation_result, group_result, filters_applied, intent,
      confidence, warnings, execution_time_ms, cached
  - AggregationResult imported from query_plan_schema (single source of truth)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from schemas.query_plan_schema import AggregationResult


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Returned after a successful Excel file upload."""

    success: bool = True
    fileId: str = Field(..., description="Unique identifier for the uploaded file session")
    sheetName: str = Field(..., description="Name of the first (active) sheet")
    totalRows: int = Field(..., description="Number of data rows (excluding header)")
    totalColumns: int = Field(..., description="Number of columns in the sheet")
    headers: list[str] = Field(..., description="List of column header names")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class ExcelQueryRequest(BaseModel):
    """Request body for the natural-language query endpoint."""

    fileId: str = Field(..., description="The fileId returned from /upload")
    question: str = Field(
        ...,
        min_length=3,
        description="Natural-language question about the Excel data",
    )


class ExcelQueryResponse(BaseModel):
    """
    Response returned after AI-powered data querying.

    Core fields (always present):
      success, summary, count, table, intent

    Quality fields:
      confidence   — AI confidence score (0.0–1.0)
      warnings     — Non-fatal issues (fuzzy column matches, type mismatches, etc.)
      filters_applied — Exact filter conditions executed (audit trail)

    Aggregation fields (present when intent is aggregate/group):
      aggregation_result — Scalar result (e.g., SUM(Outstanding Amount) = 4,500,000)
      group_result       — Per-group breakdown (e.g., by Branch)

    Performance fields:
      execution_time_ms — Total query execution time in milliseconds
      cached            — True if this result used a cached query plan (no Gemini call)
    """

    success: bool = True

    # --- Core ---
    summary: str = Field(..., description="Human-readable AI summary of the result")
    count: int = Field(..., description="Number of matching rows in the table")
    table: list[dict[str, Any]] = Field(..., description="Matching rows as list of row-dicts")
    intent: str = Field(..., description="Query intent: filter | aggregate | group | rank")

    # --- Quality ---
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="AI confidence score (0.0 = uncertain, 1.0 = certain)",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings: fuzzy column matches, type mismatches, etc.",
    )
    filters_applied: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Exact filter conditions executed — for audit and verification",
    )

    # --- Aggregation (optional) ---
    aggregation_result: Optional[AggregationResult] = Field(
        None,
        description="Scalar aggregation result (e.g., SUM / COUNT / AVG)",
    )
    group_result: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Per-group breakdown sorted by aggregated value (descending)",
    )

    # --- Performance ---
    execution_time_ms: Optional[float] = Field(
        None,
        description="Total query time in milliseconds (AI parse + Pandas execution + summary)",
    )
    cached: bool = Field(
        False,
        description="True if the query plan was served from cache (no Gemini API call made)",
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


class SaveReportRequest(BaseModel):
    """Request body for saving a query result to MongoDB."""

    query: str = Field(..., description="Label or description for this report")
    rows: list[dict[str, Any]] = Field(
        ..., description="The rows to persist (as returned in ExcelQueryResponse.table)"
    )


class SaveReportResponse(BaseModel):
    """Returned after a successful report save."""

    success: bool = True
    message: str = "Report saved successfully."


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class HistoryResponse(BaseModel):
    """One entry in the saved-report history list."""

    id: str = Field(..., description="MongoDB document _id as string")
    query: str = Field(..., description="The label saved with this report")
    createdAt: datetime = Field(..., description="UTC timestamp of when the report was saved")
    totalRows: int = Field(..., description="Number of rows saved in this report")
    rows: Optional[list[dict[str, Any]]] = Field(None, description="The rows saved in this report")


# ---------------------------------------------------------------------------
# Table Filter  (POST /api/excel/filter-table)
# ---------------------------------------------------------------------------


class ColumnFilterSpec(BaseModel):
    """
    One filter condition applied directly on a column — no AI involved.

    Operators supported (same as query engine):
      ==, !=, >, >=, <, <=          — equality / numeric
      contains, not_contains         — partial text match
      startswith, endswith           — prefix / suffix
      in, not_in                     — value in list
      isnull, notnull                — null / non-null check
      between                        — value is [low, high]
      month, year                    — date filters
      this_week, last_week, this_month, last_month — relative date windows
    """

    column: str = Field(..., description="Column name to filter on")
    operator: str = Field(..., description="Filter operator (==, contains, >, in, month, ...)")
    value: Any = Field(
        None,
        description="Filter value. List for 'in'/'between'. null for isnull/notnull/this_week etc.",
    )


class SortSpec(BaseModel):
    """Sort direction for a column."""

    column: str = Field(..., description="Column to sort by")
    ascending: bool = Field(True, description="True = A→Z / 0→9, False = Z→A / 9→0")


class TableFilterRequest(BaseModel):
    """
    Request body for POST /api/excel/filter-table.

    No AI is involved — this is a pure Pandas operation.

    Workflow:
      1. User receives a query result table from /query
      2. User clicks column filter, searches, reorders, or paginates in the UI
      3. Frontend calls /filter-table with the updated filter state
      4. Server returns filtered + sorted + paginated rows instantly

    fields:
      fileId          — the same fileId from /upload
      filters         — list of column filter conditions (AND logic between them)
      visible_columns — if set, only these columns are returned (column picker)
      search          — global text search across all visible columns (case-insensitive)
      sort_by         — sort by a column (ascending or descending)
      page            — 1-indexed page number (default: 1)
      page_size       — rows per page (default: 50, max: 500)
    """

    fileId: str = Field(..., description="The fileId returned from /upload")
    filters: list[ColumnFilterSpec] = Field(
        default_factory=list,
        description="Column filter conditions — all applied with AND logic",
    )
    visible_columns: Optional[list[str]] = Field(
        None,
        description="Columns to include in the response. None = all columns.",
    )
    search: Optional[str] = Field(
        None,
        min_length=1,
        description="Global text search — matches rows where ANY visible column contains this text",
    )
    sort_by: Optional[SortSpec] = Field(
        None,
        description="Sort the result by a column",
    )
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(50, ge=1, le=500, description="Rows per page (max 500)")


class ColumnStats(BaseModel):
    """Unique value statistics for a single column — used to build filter dropdowns in the UI."""

    column: str
    dtype: str = Field(..., description="Inferred type: numeric | date | string | boolean")
    unique_count: int
    unique_values: list[Any] = Field(
        ...,
        description="Up to 50 unique values for building filter dropdowns",
    )
    min_value: Any = None
    max_value: Any = None
    null_count: int = 0


class TableFilterResponse(BaseModel):
    """
    Response for POST /api/excel/filter-table.

    Pagination:
      total_count — total matching rows BEFORE pagination (for page count calculation)
      page        — current page
      page_size   — rows per page
      total_pages — ceil(total_count / page_size)

    Table:
      columns — column names in the response (may be subset if visible_columns was set)
      table   — current page's rows as list of dicts

    Performance:
      execution_time_ms — pure Pandas time (no AI — should be <50ms for most datasets)
    """

    success: bool = True

    # Pagination metadata
    total_count: int = Field(..., description="Total rows matching all filters (before pagination)")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Rows per page")
    total_pages: int = Field(..., description="Total number of pages")

    # Result
    columns: list[str] = Field(..., description="Column names in this response")
    table: list[dict[str, Any]] = Field(..., description="Rows for the current page")

    # Metadata
    filters_applied: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Exact filter conditions that were applied",
    )
    search_applied: Optional[str] = Field(None, description="Global search term if applied")
    sort_applied: Optional[dict[str, Any]] = Field(None, description="Sort spec if applied")
    execution_time_ms: float = Field(..., description="Pure Pandas execution time in milliseconds")


class ColumnStatsRequest(BaseModel):
    """Request body for POST /api/excel/column-stats."""

    fileId: str = Field(..., description="The fileId returned from /upload")
    columns: Optional[list[str]] = Field(
        None,
        description="Columns to return stats for. None = all columns.",
    )


class ColumnStatsResponse(BaseModel):
    """
    Response for POST /api/excel/column-stats.

    Used by the frontend to build filter dropdowns, search bars,
    and column type icons in the table header.
    """

    success: bool = True
    fileId: str
    total_rows: int
    stats: list[ColumnStats]
