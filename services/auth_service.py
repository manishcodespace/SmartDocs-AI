"""
services/auth_service.py

Responsibilities:
  - Hash & verify passwords using bcrypt (passlib)
  - Create & decode JWT access tokens (python-jose)
  - Store and retrieve users from MongoDB (collection: "users")

MongoDB 'users' collection schema:
    {
        "_id"            : ObjectId,
        "username"       : str   (unique, lowercase),
        "hashed_password": str   (bcrypt hash),
        "createdAt"      : datetime (UTC)
    }

This module is completely independent from the existing RAG / PDF services.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing  (uses bcrypt directly — no passlib needed)
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """Return a bcrypt hash of *plain_password*."""
    import bcrypt
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if *plain_password* matches *hashed_password*."""
    import bcrypt
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

_ALGORITHM = "HS256"


def _get_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set in the environment. "
            "Add it to your .env file."
        )
    return secret


def _get_expire_hours() -> int:
    try:
        return int(os.getenv("JWT_EXPIRE_HOURS", "24"))
    except ValueError:
        return 24


def create_access_token(user_id: str, username: str) -> tuple[str, int]:
    """
    Create a signed JWT token for the given user.

    Returns
    -------
    (token_string, expires_in_seconds)
    """
    from jose import jwt

    expire_hours = _get_expire_hours()
    expire_seconds = expire_hours * 3600
    expire_at = datetime.now(tz=timezone.utc) + timedelta(hours=expire_hours)

    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire_at,
        "iat": datetime.now(tz=timezone.utc),
    }

    token = jwt.encode(payload, _get_secret(), algorithm=_ALGORITHM)
    logger.debug("JWT created for user '%s', expires in %dh", username, expire_hours)
    return token, expire_seconds


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT token.

    Returns the payload dict on success.
    Raises ValueError with a descriptive message on failure.
    """
    from jose import ExpiredSignatureError, JWTError, jwt

    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[_ALGORITHM])
        return payload
    except ExpiredSignatureError:
        raise ValueError("Token has expired. Please log in again.")
    except JWTError as exc:
        raise ValueError(f"Invalid token: {exc}")


# ---------------------------------------------------------------------------
# MongoDB user CRUD
# ---------------------------------------------------------------------------

_USERS_COLLECTION = "users"


def _get_users_collection():
    """Return the MongoDB 'users' collection, creating a unique index if needed."""
    from pymongo import MongoClient, ASCENDING

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name   = os.getenv("MONGO_DB_NAME", "smartdocs_ai")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    col = db[_USERS_COLLECTION]

    # Ensure username is unique (idempotent — safe to call repeatedly)
    col.create_index([("username", ASCENDING)], unique=True)
    return col


def create_user(username: str, plain_password: str) -> dict[str, Any]:
    """
    Create a new user in MongoDB.

    Raises
    ------
    ValueError  if the username already exists.
    RuntimeError on unexpected DB errors.
    """
    from pymongo.errors import DuplicateKeyError

    try:
        col = _get_users_collection()
        hashed = hash_password(plain_password)

        doc = {
            "username":        username.lower().strip(),
            "hashed_password": hashed,
            "createdAt":       datetime.now(tz=timezone.utc),
        }

        result = col.insert_one(doc)
    except DuplicateKeyError:
        raise ValueError(f"Username '{username}' is already taken. Please choose another.")
    except Exception as exc:
        logger.exception("Unexpected error creating user '%s'.", username)
        raise RuntimeError(f"Could not create user (check DB connection): {exc}") from exc

    user_id = str(result.inserted_id)
    logger.info("New user registered: username='%s' | id=%s", username, user_id)

    return {
        "id":        user_id,
        "username":  doc["username"],
        "createdAt": doc["createdAt"],
    }


def get_user_by_username(username: str) -> dict[str, Any] | None:
    """
    Fetch a user document by username.
    Returns None if not found.
    """
    try:
        col = _get_users_collection()
        doc = col.find_one({"username": username.lower().strip()})
    except Exception as exc:
        logger.exception("Database connection error fetching user '%s'.", username)
        raise RuntimeError(f"Database query failed: {exc}") from exc

    if doc is None:
        return None
    doc["id"] = str(doc.pop("_id"))
    return doc


def authenticate_user(username: str, plain_password: str) -> dict[str, Any]:
    """
    Verify credentials and return the user document.

    Raises
    ------
    ValueError  if username not found or password is wrong.
    """
    user = get_user_by_username(username)
    if user is None:
        raise ValueError("Invalid username or password.")

    if not verify_password(plain_password, user["hashed_password"]):
        raise ValueError("Invalid username or password.")

    logger.info("User authenticated: username='%s'", username)
    return user
