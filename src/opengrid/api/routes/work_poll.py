"""
Work polling endpoint — allows NAT'd worker nodes to pull work from
the coordinator instead of the coordinator pushing to them.

This inverts the connection direction: worker connects outbound to
coordinator (works through NAT), asks "got any work for me?",
processes it locally, and sends the result back.

GET  /v1/work/poll?node_id=xxx          — grab next available job
POST /v1/work/result                    — submit completed result
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request, HTTPException
from pydantic import BaseModel

from opengrid.api.middleware.auth import verify_api_key

router = APIRouter()


class WorkPollResponse(BaseModel):
    has_work: bool = False
    job_id: str = ""
    model_id: str = ""
    prompt: str = ""
    max_tokens: int = 512
    temperature: float = 0.7


class WorkResultSubmission(BaseModel):
    job_id: str
    node_id: str
    output_text: str
    tokens_generated: int = 0
    latency_ms: float = 0.0
    error: str = ""


# Simple in-memory work queue
_pending_jobs: list[dict] = []
_completed_jobs: dict[str, WorkResultSubmission] = {}
_waiting_events: dict[str, asyncio.Event] = {}


def enqueue_job(job_id: str, model_id: str, prompt: str,
                max_tokens: int = 512, temperature: float = 0.7) -> asyncio.Event:
    """Add a job to the queue. Returns an event that fires when the result arrives."""
    event = asyncio.Event()
    _pending_jobs.append({
        "job_id": job_id,
        "model_id": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "queued_at": time.time(),
    })
    _waiting_events[job_id] = event
    return event


def get_result(job_id: str) -> Optional[WorkResultSubmission]:
    return _completed_jobs.pop(job_id, None)


@router.get("/v1/work/poll", dependencies=[Depends(verify_api_key)])
async def poll_work(node_id: str, request: Request):
    """
    Worker calls this to grab the next available job.
    Returns has_work=false if queue is empty.
    """
    if not _pending_jobs:
        return WorkPollResponse(has_work=False)

    job = _pending_jobs.pop(0)
    return WorkPollResponse(
        has_work=True,
        job_id=job["job_id"],
        model_id=job["model_id"],
        prompt=job["prompt"],
        max_tokens=job["max_tokens"],
        temperature=job["temperature"],
    )


@router.post("/v1/work/result", dependencies=[Depends(verify_api_key)])
async def submit_result(result: WorkResultSubmission, request: Request):
    """
    Worker submits completed inference result.
    """
    _completed_jobs[result.job_id] = result
    event = _waiting_events.pop(result.job_id, None)
    if event:
        event.set()

    # Credit the worker
    ledger = request.app.state.ledger
    if ledger and not result.error:
        ledger.record_earned(
            job_id=result.job_id,
            node_id=result.node_id,
            model_id="remote",
            shard_range=(0, 0),
            tokens=result.tokens_generated,
        )

    return {"status": "accepted", "job_id": result.job_id}
