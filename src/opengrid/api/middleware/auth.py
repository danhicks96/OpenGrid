"""API key authentication middleware."""
from __future__ import annotations

import os
from fastapi import Header, HTTPException, status
from typing import Optional

_API_KEY = os.environ.get("OPENGRID_API_KEY", "dev-key-change-me")


async def verify_api_key(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    scheme, _, key = authorization.partition(" ")
    if scheme.lower() != "bearer" or key != _API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return key
