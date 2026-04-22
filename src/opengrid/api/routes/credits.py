"""GET /v1/credits/balance — credit balance endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from opengrid.api.middleware.auth import verify_api_key

router = APIRouter()


@router.get("/v1/credits/balance", dependencies=[Depends(verify_api_key)])
async def credit_balance(request: Request):
    ledger = request.app.state.ledger
    balance = ledger.balance() if ledger else 0.0
    return {"balance_it": round(balance, 4), "unit": "inference_tokens"}


@router.get("/v1/network/status")
async def network_status(request: Request):
    gossip = request.app.state.gossip
    peers = gossip.active_peers() if gossip else []
    return {
        "status": "ok",
        "active_peers": len(peers),
        "peers": [
            {"node_id": p.node_id, "tier": p.tier, "jobs_active": p.jobs_active}
            for p in peers
        ],
    }
