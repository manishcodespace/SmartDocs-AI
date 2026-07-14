"""
schemas/excel_schema.py

Pydantic models for the Excel AI Query module.
Completely independent from the existing PDF/RAG schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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
    """Response returned after AI-powered data filtering."""

    success: bool = True
    summary: str = Field(..., description="Human-readable AI summary of the result")
    count: int = Field(..., description="Number of matching rows")
    table: list[dict[str, Any]] = Field(
        ..., description="Matching rows as a list of row-dicts"
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
