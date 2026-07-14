"""
dependencies/auth.py

FastAPI dependency — get_current_user

Usage in any protected route:
    from dependencies.auth import get_current_user
    from schemas.auth_schema import CurrentUser

    @router.post("/upload")
    async def upload(
        ...,
        current_user: CurrentUser = Depends(get_current_user),
    ):
        ...

The dependency:
  1. Extracts the Bearer token from the Authorization header
  2. Decodes & validates the JWT
  3. Fetches the user from MongoDB
  4. Returns a CurrentUser model
  5. Raises HTTP 401 on any failure
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from schemas.auth_schema import CurrentUser
from services import auth_service

logger = logging.getLogger(__name__)

# FastAPI reads the token from:  Authorization: Bearer <token>
# tokenUrl is used by Swagger UI's "Authorize" button
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    """
    Validate the JWT token and return the authenticated user.
    Raises HTTP 401 if the token is missing, invalid, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1. Decode JWT
    try:
        payload = auth_service.decode_access_token(token)
    except ValueError as exc:
        logger.warning("Token decode failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str | None  = payload.get("sub")
    username: str | None = payload.get("username")

    if not user_id or not username:
        raise credentials_exception

    # 2. Confirm user still exists in MongoDB
    try:
        user_doc = auth_service.get_user_by_username(username)
    except Exception as exc:
        logger.error("DB error while verifying user '%s': %s", username, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable.",
        )

    if user_doc is None:
        raise credentials_exception

    return CurrentUser(id=user_id, username=username)
