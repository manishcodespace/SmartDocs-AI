"""
routers/excel.py  (enterprise upgrade)

Excel AI Query — FastAPI router.

Routes:
    GET   /api/excel/health   — health check: server, MongoDB, Gemini, cache stats (public)
    POST  /api/excel/upload   — upload Excel file, load into memory, register schema (protected)
    POST  /api/excel/query    — natural-language query with full enterprise pipeline (protected)
    POST  /api/excel/save     — persist a query result to MongoDB (protected)
    GET   /api/excel/history  — list previously saved reports (protected)

Enterprise query pipeline (POST /query):
    ┌─────────────────────────────────────────────────────────────┐
    │ User question + fileId                                      │
    │         ↓                                                   │
    │ [Query Plan Cache] ──── HIT ──→ skip AI, use cached plan   │
    │         │ MISS                                              │
    │         ↓                                                   │
    │ [Schema Registry]  ──────────→ column types + samples      │
    │         ↓                                                   │
    │ [AI Intent Parser] ──────────→ raw intent JSON (Gemini)    │
    │         ↓                                                   │
    │ [Query Planner]    ──────────→ typed ExecutionPlan          │
    │         ↓                                                   │
    │ [Query Validator]  ──────────→ fuzzy resolve + confidence   │
    │         ↓                  └─→ warnings + errors            │
    │ [Cache PUT]                                                 │
    │         ↓                                                   │
    │ [Pandas: Filters]  ──────────→ filtered DataFrame           │
    │ [Pandas: Aggregate]──────────→ scalar result (if needed)    │
    │ [Pandas: Group By] ──────────→ per-group table (if needed)  │
    │ [Pandas: Sort/Rank]──────────→ sorted/limited rows          │
    │         ↓                                                   │
    │ [LLM Summarizer]   ──────────→ rich summary (real values)  │
    │         ↓                                                   │
    │ ExcelQueryResponse {summary, count, table, confidence, ...} │
    └─────────────────────────────────────────────────────────────┘

This router is completely independent from the existing PDF/RAG routes.
All business logic lives in the services/ layer (thin router principle).
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time

import pandas as pd
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
from services import (
    excel_ai_agent,
    excel_loader,
    mongodb_service,
)
from services import excel_aggregation_service as agg_service
from services import excel_query_service as query_service
from services import query_plan_cache
from services import query_planner
from services import query_validator
from services import schema_registry

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    summary="Health check",
    description=(
        "Returns live status of the Excel AI Query module including "
        "MongoDB, Gemini API key, file store, schema registry, and query plan cache."
    ),
    status_code=status.HTTP_200_OK,
)
async def health_check() -> dict:
    from datetime import datetime, timezone

    result: dict = {
        "status": "ok",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "module": "Excel AI Query — Enterprise Edition",
        "checks": {},
    }

    # --- Gemini API key ---
    has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    result["checks"]["gemini_api_key"] = (
        {"status": "ok", "message": "API key is set"}
        if has_key
        else {"status": "error", "message": "API key is missing"}
    )

    # --- MongoDB ---
    try:
        from pymongo import MongoClient

        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGO_DB_NAME", "smartdocs_ai")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        client.close()
        result["checks"]["mongodb"] = {
            "status": "ok",
            "message": f"Connected to '{db_name}'",
        }
    except Exception as exc:
        result["checks"]["mongodb"] = {"status": "error", "message": str(exc)}
        result["status"] = "degraded"

    # --- File store ---
    active_sessions = excel_loader.store_size()
    result["checks"]["file_store"] = {
        "status": "ok",
        "active_sessions": active_sessions,
        "message": f"{active_sessions} file(s) currently loaded in memory",
    }

    # --- Schema registry ---
    result["checks"]["schema_registry"] = {
        "status": "ok",
        "schemas_registered": schema_registry.registry_size(),
    }

    # --- Query plan cache ---
    cache_stats = query_plan_cache.cache_stats()
    result["checks"]["query_plan_cache"] = {
        "status": "ok",
        **cache_stats,
    }

    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _require_api_key() -> None:
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
        "Upload a .xlsx or .xls file. The file is read into memory, "
        "column schema is registered, and a unique fileId is returned for subsequent queries."
    ),
    status_code=status.HTTP_200_OK,
)
async def upload_excel(
    file: UploadFile = File(..., description="Excel file (.xlsx or .xls)"),
    current_user: CurrentUser = Depends(get_current_user),
) -> UploadResponse:
    tmp_path: str | None = None

    # Validate extension
    try:
        excel_loader.validate_extension(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Validate file size via headers
    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds the 10 MB limit.",
        )

    # Save to temp file
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

    # Validate file size on disk
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
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
        "Upload OK: fileId=%s | sheet=%s | rows=%d",
        metadata["fileId"], metadata["sheetName"], metadata["totalRows"],
    )
    return UploadResponse(**metadata)


# ---------------------------------------------------------------------------
# POST /query  — Enterprise pipeline
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=ExcelQueryResponse,
    summary="Query Excel data with natural language",
    description=(
        "Send a fileId (from /upload) and a natural-language question. "
        "The enterprise pipeline: Schema Registry → Cache check → AI Intent Parse → "
        "Query Planner → Query Validator (fuzzy resolve + confidence) → "
        "Pandas Execute (filters + aggregation + group-by + sort/rank) → LLM Summarizer."
    ),
    status_code=status.HTTP_200_OK,
)
async def query_excel(
    body: ExcelQueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
) -> ExcelQueryResponse:
    _require_api_key()

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Retrieve cached DataFrame
    # ------------------------------------------------------------------
    try:
        df = excel_loader.get_dataframe(body.fileId)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # 2. Get schema from registry
    # ------------------------------------------------------------------
    try:
        schema = schema_registry.get_schema(body.fileId)
    except KeyError:
        # Fallback: schema not found (e.g., server restart) — use column names only
        logger.warning("Schema not found for fileId=%s — using column names only", body.fileId)
        schema = body.fileId  # will trigger fallback in excel_ai_agent._format_schema

    # ------------------------------------------------------------------
    # 3. Check query plan cache
    # ------------------------------------------------------------------
    cached_plan = query_plan_cache.get(body.fileId, body.question)
    was_cached = cached_plan is not None

    if was_cached:
        plan = cached_plan
        # Re-validate to get fresh confidence/warnings (schema may have same structure)
        validation = query_validator.validate(plan, schema)
        logger.info("Query plan served from cache for fileId=%s", body.fileId)
    else:
        # ------------------------------------------------------------------
        # 4. AI intent parse (Gemini, temperature=0)
        # ------------------------------------------------------------------
        try:
            ai_result = excel_ai_agent.parse_question(
                question=body.question,
                schema=schema,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc
        except Exception as exc:
            _handle_gemini_error(exc)

        logger.info(
            "AI intent: intent=%s | filters=%d | q='%s...'",
            ai_result.get("intent", "?"),
            len(ai_result.get("filters", [])),
            body.question[:50],
        )

        # ------------------------------------------------------------------
        # 5. Query Planner: AI dict → typed ExecutionPlan
        # ------------------------------------------------------------------
        plan = query_planner.build_plan(ai_result)

        # ------------------------------------------------------------------
        # 6. Query Validator: fuzzy resolve columns + type checks + confidence
        # ------------------------------------------------------------------
        validation = query_validator.validate(plan, schema)
        plan = validation.resolved_plan  # use column-name-corrected plan

        # ------------------------------------------------------------------
        # 7. Cache the validated plan (if valid)
        # ------------------------------------------------------------------
        if validation.is_valid:
            query_plan_cache.put(body.fileId, body.question, plan)

    # ------------------------------------------------------------------
    # 8. Execute filters (Pandas)
    # ------------------------------------------------------------------
    filter_dicts = [f.model_dump() for f in plan.filters]
    result_df, count = query_service.apply_filters(df, filter_dicts)

    # ------------------------------------------------------------------
    # 9. Scalar aggregation (Pandas) — if intent is aggregate
    # ------------------------------------------------------------------
    agg_result = None
    if plan.aggregation:
        try:
            agg_result = agg_service.aggregate(result_df, plan.aggregation)
        except Exception as exc:
            logger.warning("Aggregation failed: %s", exc)
            validation.warnings.append(f"Aggregation could not be computed: {exc}")

    # ------------------------------------------------------------------
    # 10. Group-by aggregation (Pandas) — if intent is group
    # ------------------------------------------------------------------
    group_result = None
    if plan.group_by:
        try:
            group_result = agg_service.group_and_aggregate(result_df, plan.group_by)
        except Exception as exc:
            logger.warning("Group-by failed: %s", exc)
            validation.warnings.append(f"Group-by could not be computed: {exc}")

    # ------------------------------------------------------------------
    # 11. Sort / Rank (Pandas) — if sort_by or intent is rank
    # ------------------------------------------------------------------
    rows: list[dict]
    if plan.sort_by:
        sorted_df = agg_service.sort_and_rank(result_df, plan.sort_by, plan.limit)
        rows = query_service.df_to_records(sorted_df)
        count = len(rows)
    elif group_result is not None:
        # Return group result as the table for group-by queries
        rows = group_result
        count = len(rows)
    elif plan.limit:
        all_rows = query_service.df_to_records(result_df)
        rows = all_rows[: plan.limit]
        count = len(rows)
    else:
        rows = query_service.df_to_records(result_df)

    # ------------------------------------------------------------------
    # 12. Build rich summary (uses REAL computed values — no hallucination)
    # ------------------------------------------------------------------
    summary = excel_ai_agent.build_rich_summary(
        template=plan.summary_template,
        count=count,
        agg_result=agg_result,
        group_result=group_result,
        intent=plan.intent,
    )

    exec_ms = round((time.perf_counter() - t_start) * 1000, 2)

    logger.info(
        "Query complete | intent=%s | rows=%d | conf=%.3f | cached=%s | time=%.1fms | q='%s...'",
        plan.intent, count, validation.confidence, was_cached, exec_ms, body.question[:50],
    )

    return ExcelQueryResponse(
        success=True,
        summary=summary,
        count=count,
        table=rows,
        intent=plan.intent,
        confidence=validation.confidence,
        warnings=validation.warnings,
        filters_applied=filter_dicts,
        aggregation_result=agg_result,
        group_result=group_result if group_result is not None else None,
        execution_time_ms=exec_ms,
        cached=was_cached,
    )


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
        inserted_id = mongodb_service.save_report(query=body.query, rows=body.rows, user_id=current_user.id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error saving report to MongoDB.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save the report: {exc}",
        ) from exc

    logger.info("Report saved: id=%s | query='%s'", inserted_id, body.query)
    return SaveReportResponse(success=True, message="Report saved successfully.")


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------


@router.get(
    "/history",
    response_model=list[HistoryResponse],
    summary="Retrieve saved report history",
    description="Return a summary list of all previously saved reports (newest first) or filter by a specific user ID.",
    status_code=status.HTTP_200_OK,
)
async def get_history(
    id: str | None = None,
    current_user: CurrentUser = Depends(get_current_user),
) -> list[HistoryResponse]:
    try:
        # Use query param 'id' as the user_id filter if provided, otherwise default to current_user.id
        records = mongodb_service.get_history(user_id=id or current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
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
