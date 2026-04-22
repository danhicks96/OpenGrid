"""
POST /v1/chat/completions — OpenAI-compatible chat completions endpoint.
Supports streaming via Server-Sent Events.
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


async def _stream_response(request: Request, req: ChatCompletionRequest) -> AsyncIterator[str]:
    scheduler = request.app.state.scheduler
    ledger = request.app.state.ledger

    result = scheduler.schedule(req.model, session_id=req.session_id)
    if result.dag is None:
        yield f"data: {json.dumps({'error': result.error})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Stub: yield a placeholder response until inference is wired end-to-end
    stub_reply = (
        "[OpenGrid stub] Inference pipeline scheduled. "
        f"DAG: {len(result.dag.tasks)} stage(s) across "
        f"{len(set(t.node_id for t in result.dag.tasks))} node(s). "
        "Connect worker backends to receive real model output."
    )
    for word in stub_reply.split():
        yield _sse_chunk(word + " ")
        await asyncio.sleep(0.02)

    if ledger:
        ledger.record_spent(
            job_id=result.dag.job_id,
            node_id="local",
            model_id=req.model,
            tokens=len(stub_reply.split()),
        )

    yield _sse_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatCompletionRequest, request: Request):
    await require_positive_balance(request)

    if req.stream:
        return StreamingResponse(
            _stream_response(request, req),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks
    chunks = []
    async for chunk in _stream_response(request, req):
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            try:
                data = json.loads(chunk[6:])
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks.append(delta)
            except Exception:
                pass

    content = "".join(chunks)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
            "index": 0,
        }],
        "usage": {
            "prompt_tokens": sum(len(m.content.split()) for m in req.messages),
            "completion_tokens": len(content.split()),
            "total_tokens": sum(len(m.content.split()) for m in req.messages) + len(content.split()),
        },
    }
