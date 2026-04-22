"""GET /v1/models — lists available models."""
from __future__ import annotations

import time
from fastapi import APIRouter, Depends, Request
from opengrid.api.middleware.auth import verify_api_key

router = APIRouter()


@router.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models(request: Request):
    registry = request.app.state.model_registry
    models = registry.list_models() if registry else []
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": int(time.time()), "owned_by": "opengrid"}
            for m in models
        ],
    }
