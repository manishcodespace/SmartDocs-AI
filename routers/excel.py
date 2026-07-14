"""
routers/excel.py

Excel AI Query — FastAPI router.

Routes:
    GET   /api/excel/health   — check server + MongoDB + Gemini API key status (public)
    POST  /api/excel/upload   — upload an Excel file, store in memory            (protected)
    POST  /api/excel/query    — natural-language query against uploaded file     (protected)
    POST  /api/excel/save     — persist a query result to MongoDB                (protected)
    GET   /api/excel/history  — list previously saved reports                   (protected)

This router is completely independent from the existing PDF/RAG routes.
All business logic lives in the services/ layer.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from dependencies.auth import get_current_user
from schemas.auth_schema import CurrentUser
from schemas.excel_schema import (
    ExcelQueryRequest,
    ExcelQueryResponse,
    HistoryResponse,
    SaveReportRequest,
    SaveReportResponse,
    UploadResponse,
)
from services import excel_ai_agent, excel_loader, excel_query_service, mongodb_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum file size: 10 MB
_MAX_FILE_SIZE = 10 * 1024 * 1024


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    summary="Health check",
    description=(
        "Returns the live status of the Excel AI Query module, "
        "MongoDB Atlas connection, Gemini API key, and active file sessions."
    ),
    status_code=status.HTTP_200_OK,
)
async def health_check() -> dict:
    import os
    from datetime import datetime, timezone

    result: dict = {
        "status": "ok",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "module": "Excel AI Query",
        "checks": {},
    }

    # --- 1. Gemini API key ---
    has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    result["checks"]["gemini_api_key"] = (
        {"status": "ok", "message": "API key is set"}
        if has_key
        else {"status": "error", "message": "API key is missing"}
    )

    # --- 2. MongoDB Atlas ---
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name   = os.getenv("MONGO_DB_NAME", "smartdocs_ai")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        client.close()
        result["checks"]["mongodb"] = {
            "status": "ok",
            "message": f"Connected to '{db_name}'",
        }
    except Exception as exc:
        result["checks"]["mongodb"] = {
            "status": "error",
            "message": str(exc),
        }
        result["status"] = "degraded"

    # --- 3. In-memory file store ---
    active_sessions = excel_loader.store_size()
    result["checks"]["file_store"] = {
        "status": "ok",
        "active_sessions": active_sessions,
        "message": f"{active_sessions} file(s) currently loaded in memory",
    }

    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _require_api_key() -> None:
    """Raise HTTP 500 if no Gemini API key is configured."""
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Google Gemini API key is missing. "
                "Please set GEMINI_API_KEY or GOOGLE_API_KEY in the environment."
            ),
        )


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload an Excel file",
    description=(
        "Upload a .xlsx or .xls file. The file is read into memory and a "
        "unique fileId is returned for subsequent queries."
    ),
    status_code=status.HTTP_200_OK,
)
async def upload_excel(
    file: UploadFile = File(..., description="Excel file (.xlsx or .xls)"),
    current_user: CurrentUser = Depends(get_current_user),
) -> UploadResponse:
    tmp_path: str | None = None

    # --- Validate extension ---
    try:
        excel_loader.validate_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # --- Validate file size (headers) ---
    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds the 10 MB limit.",
        )

    # --- Save to a temporary file ---
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save the uploaded file: {exc}",
        ) from exc

    # --- Validate file size on disk ---
    if os.path.getsize(tmp_path) > _MAX_FILE_SIZE:
        _cleanup(tmp_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds the 10 MB limit.",
        )

    try:
        metadata = excel_loader.load_excel(tmp_path, file.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error loading Excel file.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while reading the file: {exc}",
        ) from exc
    finally:
        _cleanup(tmp_path)

    logger.info(
        "Upload successful: fileId=%s | sheet=%s | rows=%d",
        metadata["fileId"],
        metadata["sheetName"],
        metadata["totalRows"],
    )

    return UploadResponse(**metadata)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=ExcelQueryResponse,
    summary="Query Excel data with natural language",
    description=(
        "Send a fileId (from /upload) and a natural-language question. "
        "The AI extracts filter conditions, applies them to the DataFrame, "
        "and returns matching rows with a summary."
    ),
    status_code=status.HTTP_200_OK,
)
async def query_excel(
    body: ExcelQueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> ExcelQueryResponse:
    _require_api_key()

    # --- Retrieve cached DataFrame ---
    try:
        df = excel_loader.get_dataframe(body.fileId)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    # --- AI: parse natural language → structured filters ---
    try:
        ai_result = excel_ai_agent.parse_question(
            question=body.question,
            column_names=list(df.columns),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _handle_gemini_error(exc)

    filters = ai_result.get("filters", [])
    summary_template = ai_result.get("summary_template", "matching records")

    logger.info(
        "AI extracted %d filter(s) for question: '%s'",
        len(filters),
        body.question,
    )

    # --- Apply filters ---
    rows, count = excel_query_service.apply_filters(df, filters)

    # --- Build human-readable summary ---
    summary = excel_ai_agent.build_summary(summary_template, count)

    return ExcelQueryResponse(success=True, summary=summary, count=count, table=rows)


# ---------------------------------------------------------------------------
# POST /save
# ---------------------------------------------------------------------------


@router.post(
    "/save",
    response_model=SaveReportResponse,
    summary="Save a query result to MongoDB",
    description="Persist the given rows under a query label in the 'savedReports' collection.",
    status_code=status.HTTP_201_CREATED,
)
async def save_report(
    body: SaveReportRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> SaveReportResponse:
    if not body.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' field cannot be empty.",
        )

    try:
        inserted_id = mongodb_service.save_report(
            query=body.query,
            rows=body.rows,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error saving report to MongoDB.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save the report: {exc}",
        ) from exc

    logger.info("Report saved with id=%s | query='%s'", inserted_id, body.query)
    return SaveReportResponse(success=True, message="Report saved successfully.")


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------


@router.get(
    "/history",
    response_model=list[HistoryResponse],
    summary="Retrieve saved report history",
    description="Return a summary list of all previously saved reports (newest first).",
    status_code=status.HTTP_200_OK,
)
async def get_history(
    current_user: CurrentUser = Depends(get_current_user),
) -> list[HistoryResponse]:
    try:
        records = mongodb_service.get_history()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error fetching history from MongoDB.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve history: {exc}",
        ) from exc

    return [HistoryResponse(**r) for r in records]


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _cleanup(path: str | None) -> None:
    """Silently remove a temporary file."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def _handle_gemini_error(exc: Exception) -> None:
    """Translate Gemini / network errors into appropriate HTTP responses."""
    err = str(exc)
    if "API_KEY_INVALID" in err or "api key not valid" in err.lower():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid Google Gemini API key.",
        )
    if "quota" in err.lower() or "429" in err or "resource_exhausted" in err.lower():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="API rate limit exceeded. Please try again in a few moments.",
        )
    if "service unavailable" in err.lower() or "503" in err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API service is temporarily unavailable.",
        )
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"An unexpected error occurred: {err}",
    )
