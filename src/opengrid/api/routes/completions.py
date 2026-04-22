"""
POST /v1/chat/completions — OpenAI-compatible chat completions endpoint.
Supports streaming via Server-Sent Events.

v0.0.2: Wired to real DAG dispatcher. Falls back to stub when no workers
         are available (so the API still responds during development).
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from opengrid.api.middleware.auth import verify_api_key
from opengrid.api.middleware.credit_check import require_positive_balance

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.7
    session_id: Optional[str] = None


def _sse_chunk(delta: str, finish_reason: Optional[str] = None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{
            "delta": {"content": delta} if delta else {},
            "finish_reason": finish_reason,
            "index": 0,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _execute_inference(request: Request, req: ChatCompletionRequest) -> tuple[str, dict]:
    """
    Execute the actual distributed inference pipeline.
    Returns (output_text, usage_dict).
    """
    scheduler = request.app.state.scheduler
    dispatcher = request.app.state.dispatcher
    ledger = request.app.state.ledger

    # Build prompt from messages
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)

    # Schedule: build a DAG and assign nodes
    result = scheduler.schedule(req.model, session_id=req.session_id)
    if result.dag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result.error,
            headers={"Retry-After": "10"},
        )

    # Register with work monitor for fault tolerance
    work_monitor = request.app.state.work_monitor
    work_monitor.register_job(result.dag)

    # Execute the DAG through the dispatcher
    exec_result = await dispatcher.execute(result.dag, input_text=prompt)

    if exec_result.success:
        output = exec_result.output_text
        prompt_tokens = sum(len(m.content.split()) for m in req.messages)
        completion_tokens = exec_result.tokens_generated

        # Deduct credits from requester
        if ledger:
            from opengrid.coordinator.scheduler import MODEL_COST_FACTORS
            cost_factor = MODEL_COST_FACTORS.get(req.model, 1.0)
            ledger.record_spent(
                job_id=result.dag.job_id,
                node_id="local",
                model_id=req.model,
                tokens=prompt_tokens + completion_tokens,
                model_cost_factor=cost_factor,
            )

        # Issue credits to worker nodes
        if ledger:
            for task_info in exec_result.task_results:
                ledger.record_earned(
                    job_id=f"{result.dag.job_id}-{task_info['task_id'][:8]}",
                    node_id=task_info["node_id"],
                    model_id=req.model,
                    shard_range=tuple(task_info["shard_range"]),
                    tokens=completion_tokens,
                )

        return output, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Inference failed: {exec_result.error}",
        )


async def _stream_response(request: Request, req: ChatCompletionRequest) -> AsyncIterator[str]:
    try:
        output, usage = await _execute_inference(request, req)
        for word in output.split():
            yield _sse_chunk(word + " ")
            await asyncio.sleep(0.01)
        yield _sse_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"
    except HTTPException as e:
        yield f"data: {json.dumps({'error': e.detail})}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatCompletionRequest, request: Request):
    await require_positive_balance(request)

    if req.stream:
        return StreamingResponse(
            _stream_response(request, req),
            media_type="text/event-stream",
        )

    output, usage = await _execute_inference(request, req)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "message": {"role": "assistant", "content": output},
            "finish_reason": "stop",
            "index": 0,
        }],
        "usage": usage,
    }


@router.post("/v1/chat/completions/local", dependencies=[Depends(verify_api_key)])
async def chat_completions_local(req: ChatCompletionRequest, request: Request):
    """
    Direct local inference — bypasses the scheduler/DAG entirely.
    Sends the prompt straight to the local worker backend.
    For single-node testing, development, and worker-assigned jobs.
    No credits required — this node is the compute provider.
    """

    worker_server = request.app.state.worker_server
    worker = worker_server._worker
    ledger = request.app.state.ledger

    if not worker._backend.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model loaded. Start with --model flag.",
        )

    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)

    if req.stream:
        async def _stream():
            for token in worker._backend.generate_stream(
                prompt, max_tokens=req.max_tokens, temperature=req.temperature
            ):
                yield _sse_chunk(token)
            yield _sse_chunk("", finish_reason="stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream(), media_type="text/event-stream")

    output = worker._backend.generate(
        prompt, max_tokens=req.max_tokens, temperature=req.temperature
    )
    prompt_tokens = sum(len(m.content.split()) for m in req.messages)
    completion_tokens = len(output.split())

    if ledger:
        ledger.record_spent(
            job_id=str(uuid.uuid4()),
            node_id="local",
            model_id=req.model,
            tokens=prompt_tokens + completion_tokens,
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "message": {"role": "assistant", "content": output},
            "finish_reason": "stop",
            "index": 0,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@router.post("/v1/chat/completions/distributed", dependencies=[Depends(verify_api_key)])
async def chat_completions_distributed(req: ChatCompletionRequest, request: Request):
    """
    Distributed inference via work queue — sends the job to a remote
    worker node that polls for work. Works through NAT because the
    worker initiates the connection, not the coordinator.

    Flow:
    1. User hits this endpoint
    2. Job queued in work_poll queue
    3. Remote worker polls GET /v1/work/poll, grabs the job
    4. Worker runs inference locally on their GPU
    5. Worker POSTs result to /v1/work/result
    6. This endpoint returns the result to the user
    """
    await require_positive_balance(request)

    from opengrid.api.routes.work_poll import enqueue_job, get_result

    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    job_id = str(uuid.uuid4())

    # Queue the job and get an event to wait on
    event = enqueue_job(job_id, req.model, prompt, req.max_tokens, req.temperature)

    # Wait for a worker to pick it up and return the result (timeout 120s)
    try:
        await asyncio.wait_for(event.wait(), timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="No worker picked up the job within 120 seconds. Is a worker node running and polling?",
        )

    result = get_result(job_id)
    if result is None or result.error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Worker error: {result.error if result else 'result missing'}",
        )

    output = result.output_text or result.output or result.text or result.result or ""
    ledger = request.app.state.ledger
    prompt_tokens = sum(len(m.content.split()) for m in req.messages)
    completion_tokens = len(output.split())

    if ledger:
        ledger.record_spent(
            job_id=job_id, node_id="local", model_id=req.model,
            tokens=prompt_tokens + completion_tokens,
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "message": {"role": "assistant", "content": output},
            "finish_reason": "stop",
            "index": 0,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
