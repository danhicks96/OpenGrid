"""
Worker node — receives work packets, runs inference, returns output + TOPLOC proof.

v0.0.2: Uses generate() for full-model-per-node beta mode instead of
         shard-level forward(). The worker now produces real model output.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Optional

from opengrid.daemon.shard_manager import ShardManager
from opengrid.node.inference_engine import InferenceBackend
from opengrid.node.kv_cache import KVCacheStore
from opengrid.node.toploc_prover import generate_proof

log = logging.getLogger(__name__)


@dataclass
class WorkPacket:
    packet_type: str          # "inference_forward" | "inference_generate"
    job_id: str
    sequence_id: int
    model_id: str
    shard_range: list[int]    # [lo, hi]
    input_activations_b64: str
    kv_cache_token: str
    return_address: str
    deadline_ms: int = 2000
    max_tokens: int = 512
    temperature: float = 0.7


@dataclass
class WorkResult:
    job_id: str
    sequence_id: int
    shard_range: list[int]
    output_activations_b64: str
    toploc_proof: str
    latency_ms: float
    tokens_generated: int = 0
    error: str = ""


class Worker:
    def __init__(
        self,
        node_id: str,
        shard_manager: ShardManager,
        backend: InferenceBackend,
        kv_store: KVCacheStore,
    ):
        self.node_id = node_id
        self._shards = shard_manager
        self._backend = backend
        self._kv = kv_store
        self._active_jobs: dict[str, float] = {}  # job_id → start_time
        self._model_loaded: str = ""  # currently loaded model path

    def load_model(self, model_path: str) -> None:
        """Load a model into the backend. Call before handling packets."""
        if self._model_loaded == model_path and self._backend.is_loaded():
            return  # already loaded
        if self._backend.is_loaded():
            self._backend.unload()
        self._backend.load_model(model_path)
        self._model_loaded = model_path
        log.info("Worker %s loaded model: %s", self.node_id, model_path)

    async def handle_packet(self, packet: WorkPacket) -> WorkResult:
        t0 = time.perf_counter()
        self._active_jobs[packet.job_id] = t0

        try:
            # Decode input — in full-model mode this is the prompt text
            input_bytes = base64.b64decode(packet.input_activations_b64)
            prompt = input_bytes.decode("utf-8", errors="replace")

            # Check if we have a model loaded
            if not self._backend.is_loaded():
                # Try the legacy forward() path (stub passthrough)
                try:
                    output_bytes = self._backend.forward(input_bytes, tuple(packet.shard_range))
                    output_text = output_bytes.decode("utf-8", errors="replace")
                except (NotImplementedError, RuntimeError):
                    # No model loaded, return descriptive error
                    return WorkResult(
                        job_id=packet.job_id,
                        sequence_id=packet.sequence_id,
                        shard_range=packet.shard_range,
                        output_activations_b64="",
                        toploc_proof="",
                        latency_ms=0.0,
                        error="no_model_loaded",
                    )
            else:
                # Real inference — generate text from the prompt
                output_text = self._backend.generate(
                    prompt,
                    max_tokens=packet.max_tokens,
                    temperature=packet.temperature,
                )

            output_bytes = output_text.encode("utf-8")

            # Store in KV cache
            shard_range = tuple(packet.shard_range)
            self._kv.put(packet.job_id, shard_range, output_bytes)

            # Generate TOPLOC proof over the output
            proof = generate_proof(output_bytes)
            latency_ms = (time.perf_counter() - t0) * 1000

            return WorkResult(
                job_id=packet.job_id,
                sequence_id=packet.sequence_id,
                shard_range=packet.shard_range,
                output_activations_b64=base64.b64encode(output_bytes).decode(),
                toploc_proof=proof,
                latency_ms=round(latency_ms, 2),
                tokens_generated=len(output_text.split()),
            )
        except Exception as e:
            log.exception("Error handling packet for job %s", packet.job_id)
            return WorkResult(
                job_id=packet.job_id,
                sequence_id=packet.sequence_id,
                shard_range=packet.shard_range,
                output_activations_b64="",
                toploc_proof="",
                latency_ms=0.0,
                error=str(e),
            )
        finally:
            self._active_jobs.pop(packet.job_id, None)

    def active_job_count(self) -> int:
        return len(self._active_jobs)
