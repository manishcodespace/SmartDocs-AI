"""
routers/auth.py

Authentication routes for the Excel AI Query module.

Routes:
    POST  /api/auth/register   — create a new user account
    POST  /api/auth/login      — login and receive a JWT token
    GET   /api/auth/me         — get current user info (protected)

Passwords are hashed with bcrypt.
Tokens are signed JWTs (HS256), valid for JWT_EXPIRE_HOURS (default: 24h).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from dependencies.auth import get_current_user
from schemas.auth_schema import (
    CurrentUser,
    LoginRequest,
    MeResponse,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)
from services import auth_service

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------------------


@router.post(
    "/register",
    response_model=RegisterResponse,
    summary="Register a new user account",
    description="Create a new account with a unique username and password (min 6 chars).",
    status_code=status.HTTP_201_CREATED,
)
async def register(body: RegisterRequest) -> RegisterResponse:
    """Open registration — anyone can create an account."""
    try:
        user = auth_service.create_user(
            username=body.username,
            plain_password=body.password,
        )
    except ValueError as exc:
        # Username already taken
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return RegisterResponse(username=user["username"])


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get a JWT token",
    description=(
        "Authenticate with your username and password. "
        "Returns a Bearer token to use in all protected endpoints."
    ),
    status_code=status.HTTP_200_OK,
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """
    Accepts standard OAuth2 form data (username + password fields).
    Compatible with Swagger UI's built-in 'Authorize' button.
    """
    try:
        user = auth_service.authenticate_user(
            username=form_data.username,
            plain_password=form_data.password,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    token, expires_in = auth_service.create_access_token(
        user_id=user["id"],
        username=user["username"],
    )

    logger.info("Login successful: username='%s'", user["username"])

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        username=user["username"],
    )


# ---------------------------------------------------------------------------
# GET /me  (protected — requires valid JWT)
# ---------------------------------------------------------------------------


@router.get(
    "/me",
    response_model=MeResponse,
    summary="Get current user info",
    description="Returns the profile of the currently authenticated user.",
    status_code=status.HTTP_200_OK,
)
async def me(current_user: CurrentUser = Depends(get_current_user)) -> MeResponse:
    """Protected endpoint — verifies the Bearer token and returns user info."""
    try:
        user_doc = auth_service.get_user_by_username(current_user.username)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not fetch user info: {exc}",
        ) from exc

    if user_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return MeResponse(
        id=user_doc["id"],
        username=user_doc["username"],
        createdAt=user_doc["createdAt"],
    )
