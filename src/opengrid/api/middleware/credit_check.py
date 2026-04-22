"""Pre-flight credit check middleware."""
from __future__ import annotations

from fastapi import HTTPException, status, Request

MIN_BALANCE = 0.0


async def require_positive_balance(request: Request) -> None:
    ledger = request.app.state.ledger
    if ledger is None:
        return
    balance = ledger.balance()
    if balance <= MIN_BALANCE:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient credits (balance: {balance:.1f} IT). Contribute compute to earn more.",
        )
