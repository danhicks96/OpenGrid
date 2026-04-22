"""
Worker node — receives work packets, runs the shard forward pass,
returns output activations + TOPLOC proof.
"""
from __future__ import annotations

import asyncio
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
    packet_type: str          # "inference_forward"
    job_id: str
    sequence_id: int
    model_id: str
    shard_range: list[int]    # [lo, hi]
    input_activations_b64: str
    kv_cache_token: str
    return_address: str
    deadline_ms: int = 2000


@dataclass
class WorkResult:
    job_id: str
    sequence_id: int
    shard_range: list[int]
    output_activations_b64: str
    toploc_proof: str
    latency_ms: float
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

    async def handle_packet(self, packet: WorkPacket) -> WorkResult:
        t0 = time.perf_counter()
        self._active_jobs[packet.job_id] = t0

        try:
            import base64
            input_act = base64.b64decode(packet.input_activations_b64)
            shard_range = tuple(packet.shard_range)

            # Check KV cache for this job/shard
            cached_kv = self._kv.get(packet.job_id, shard_range)

            # Run forward pass (stub: returns input unchanged if backend not wired)
            try:
                output_act = self._backend.forward(input_act, shard_range)
            except NotImplementedError:
                # Stub: pass activations through unchanged
                output_act = input_act
                log.debug("Using stub forward pass for job %s", packet.job_id)

            # Store KV output
            self._kv.put(packet.job_id, shard_range, output_act)

            proof = generate_proof(output_act)
            latency_ms = (time.perf_counter() - t0) * 1000

            return WorkResult(
                job_id=packet.job_id,
                sequence_id=packet.sequence_id,
                shard_range=packet.shard_range,
                output_activations_b64=base64.b64encode(output_act).decode(),
                toploc_proof=proof,
                latency_ms=round(latency_ms, 2),
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
