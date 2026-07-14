"""
schemas/auth_schema.py

Pydantic models for the Authentication module.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class RegisterRequest(BaseModel):
    """Request body for user registration."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Unique username (3–50 characters)",
    )
    password: str = Field(
        ...,
        min_length=6,
        description="Password (minimum 6 characters)",
    )

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Username may only contain letters, numbers, underscores, and hyphens."
            )
        return stripped.lower()

    @field_validator("password")
    @classmethod
    def password_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Password cannot be blank or whitespace only.")
        return v


class LoginRequest(BaseModel):
    """Request body for user login."""

    username: str = Field(..., description="Registered username")
    password: str = Field(..., description="Account password")


class TokenResponse(BaseModel):
    """Returned after a successful login."""

    access_token: str = Field(..., description="JWT Bearer token")
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token lifetime in seconds")
    username: str = Field(..., description="Authenticated username")


class RegisterResponse(BaseModel):
    """Returned after a successful registration."""

    success: bool = True
    message: str = "Account created successfully. You can now log in."
    username: str


class CurrentUser(BaseModel):
    """Represents the authenticated user extracted from a JWT token."""

    id: str = Field(..., description="MongoDB document _id as string")
    username: str = Field(..., description="Authenticated username")


class MeResponse(BaseModel):
    """Response for GET /api/auth/me."""

    id: str
    username: str
    createdAt: datetime
