"""
DAG executor dispatch engine — the coordinator's main loop that actually
drives distributed inference.

Takes a built DAG, connects to worker nodes via WebSocket, sends work packets,
collects results, and chains activations through the pipeline.

Without this module, the DAG is a plan that never executes.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import AsyncIterator, Optional

import websockets

from opengrid.coordinator.dag_executor import InferenceDAG, TaskStatus
from opengrid.coordinator.reputation import ReputationManager
from opengrid.mesh.peer_registry import PeerRegistry
from opengrid.node.worker import WorkPacket, WorkResult

log = logging.getLogger(__name__)

# Shared HMAC key for activation integrity (in production: per-session derived key)
_HMAC_KEY = b"opengrid-activation-integrity-v1"

CONNECT_TIMEOUT = 5.0   # seconds to establish WebSocket
MAX_RETRIES = 2


def _hmac_sign(payload: bytes) -> str:
    return hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()


def _hmac_verify(payload: bytes, signature: str) -> bool:
    expected = hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


class ExecutorResult:
    """Outcome of executing a full inference DAG."""
    def __init__(self):
        self.job_id: str = ""
        self.success: bool = False
        self.output_text: str = ""
        self.tokens_generated: int = 0
        self.total_latency_ms: float = 0.0
        self.error: str = ""
        self.task_results: list[dict] = []
        # Per-node latencies for credit/reputation tracking
        self.node_latencies: dict[str, float] = {}


class DAGDispatcher:
    """
    Executes an InferenceDAG by dispatching tasks to workers via WebSocket
    and chaining activations through the pipeline.
    """

    def __init__(
        self,
        peer_registry: PeerRegistry,
        reputation: Optional[ReputationManager] = None,
    ):
        self._peers = peer_registry
        self._reputation = reputation

    async def execute(self, dag: InferenceDAG, input_text: str = "") -> ExecutorResult:
        """
        Execute a full inference DAG end-to-end.

        1. Walk ready_tasks() on each tick
        2. Connect to worker via WebSocket, send WorkPacket
        3. Receive WorkResult with output activations
        4. Feed output as input to next stage
        5. On timeout/failure: retry with backup node (up to MAX_RETRIES)
        6. Return aggregated result
        """
        result = ExecutorResult()
        result.job_id = dag.job_id
        t0 = time.perf_counter()

        # Seed the first stage with the input text as "activations"
        initial_activations = base64.b64encode(input_text.encode()).decode()

        # Track activations flowing through the pipeline
        # task_id → base64 encoded output activations
        task_outputs: dict[str, str] = {}
        upstream_input = initial_activations

        while not dag.is_complete() and not dag.has_failures():
            ready = dag.ready_tasks()
            if not ready:
                if dag.has_failures():
                    break
                # All tasks are either RUNNING or waiting on deps
                await asyncio.sleep(0.01)
                continue

            for task in ready:
                # Determine input: from upstream task or initial input
                deps = dag.dependencies.get(task.task_id, [])
                if deps:
                    upstream_id = deps[-1]
                    input_act = task_outputs.get(upstream_id, initial_activations)
                else:
                    input_act = initial_activations

                dag.mark_running(task.task_id)

                # Dispatch to worker
                work_result = await self._dispatch_task(task, input_act, dag)

                if work_result.error:
                    # Retry logic
                    if task.retry_count < MAX_RETRIES:
                        task.retry_count += 1
                        log.warning(
                            "Task %s failed (attempt %d/%d): %s — retrying",
                            task.task_id, task.retry_count, MAX_RETRIES, work_result.error,
                        )
                        # Find backup node
                        backup = self._find_backup(task.model_id, task.shard_range, task.node_id)
                        if backup:
                            task.node_id = backup
                            dag.status[task.task_id] = TaskStatus.PENDING
                            task.status = TaskStatus.PENDING
                        else:
                            dag.mark_failed(task.task_id, error=f"no backup node: {work_result.error}")
                            if self._reputation:
                                self._reputation.timeout_fault(task.node_id)
                    else:
                        dag.mark_failed(task.task_id, error=work_result.error)
                        if self._reputation:
                            self._reputation.timeout_fault(task.node_id)
                else:
                    # Success
                    output_bytes = base64.b64decode(work_result.output_activations_b64)
                    dag.mark_done(task.task_id, result=output_bytes)
                    task_outputs[task.task_id] = work_result.output_activations_b64
                    result.node_latencies[task.node_id] = work_result.latency_ms
                    result.task_results.append({
                        "task_id": task.task_id,
                        "node_id": task.node_id,
                        "latency_ms": work_result.latency_ms,
                        "shard_range": work_result.shard_range,
                    })

                    # Update reputation
                    if self._reputation:
                        self._reputation.proof_passed(task.node_id)
                    peer_entry = self._peers.get(task.node_id)
                    if peer_entry:
                        self._peers.record_job_done(task.node_id, work_result.latency_ms)

        result.total_latency_ms = (time.perf_counter() - t0) * 1000

        if dag.is_complete():
            # Get final output from last task
            if dag.tasks:
                last_task_id = dag.tasks[-1].task_id
                final_output_b64 = task_outputs.get(last_task_id, "")
                if final_output_b64:
                    try:
                        final_bytes = base64.b64decode(final_output_b64)
                        result.output_text = final_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        result.output_text = final_output_b64
            result.success = True
            result.tokens_generated = len(result.output_text.split())
        else:
            # Find the failure
            for task in dag.tasks:
                if task.status == TaskStatus.FAILED:
                    result.error = task.error
                    break
            if not result.error:
                result.error = "DAG execution failed (unknown reason)"

        return result

    async def execute_streaming(
        self, dag: InferenceDAG, input_text: str = ""
    ) -> AsyncIterator[str]:
        """
        Execute DAG and yield tokens as they arrive.
        For full-model-per-node mode (beta), the entire generation happens
        on one node and tokens stream back.
        """
        result = await self.execute(dag, input_text)
        if result.success:
            # Yield words one at a time for streaming effect
            for word in result.output_text.split():
                yield word + " "
        else:
            yield f"[error: {result.error}]"

    async def _dispatch_task(
        self, task, input_activations_b64: str, dag: InferenceDAG
    ) -> WorkResult:
        """Connect to worker node and send a work packet."""
        peer = self._peers.get(task.node_id)
        if peer is None:
            return WorkResult(
                job_id=dag.job_id, sequence_id=0, shard_range=list(task.shard_range),
                output_activations_b64="", toploc_proof="",
                latency_ms=0.0, error=f"peer {task.node_id} not found in registry",
            )

        host = peer.record.host
        port = peer.record.port
        if not host or not port:
            return WorkResult(
                job_id=dag.job_id, sequence_id=0, shard_range=list(task.shard_range),
                output_activations_b64="", toploc_proof="",
                latency_ms=0.0, error=f"peer {task.node_id} has no address (host={host}, port={port})",
            )

        uri = f"ws://{host}:{port}"
        packet = {
            "packet_type": "inference_forward",
            "job_id": dag.job_id,
            "sequence_id": 0,
            "model_id": task.model_id,
            "shard_range": list(task.shard_range),
            "input_activations_b64": input_activations_b64,
            "kv_cache_token": f"kv-{dag.job_id}-{task.shard_range[0]}-{task.shard_range[1]}",
            "return_address": "",
            "deadline_ms": task.deadline_ms,
        }

        # Add HMAC for activation integrity
        act_bytes = base64.b64decode(input_activations_b64)
        packet["activation_hmac"] = _hmac_sign(act_bytes)

        try:
            async with websockets.connect(uri, open_timeout=CONNECT_TIMEOUT) as ws:
                await ws.send(json.dumps(packet))
                response_raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=task.deadline_ms / 1000.0 + CONNECT_TIMEOUT,
                )
                data = json.loads(response_raw)
                return WorkResult(
                    job_id=data.get("job_id", dag.job_id),
                    sequence_id=data.get("sequence_id", 0),
                    shard_range=data.get("shard_range", list(task.shard_range)),
                    output_activations_b64=data.get("output_activations_b64", ""),
                    toploc_proof=data.get("toploc_proof", ""),
                    latency_ms=data.get("latency_ms", 0.0),
                    error=data.get("error", ""),
                )
        except asyncio.TimeoutError:
            return WorkResult(
                job_id=dag.job_id, sequence_id=0, shard_range=list(task.shard_range),
                output_activations_b64="", toploc_proof="",
                latency_ms=float(task.deadline_ms),
                error=f"timeout connecting to {uri}",
            )
        except Exception as e:
            return WorkResult(
                job_id=dag.job_id, sequence_id=0, shard_range=list(task.shard_range),
                output_activations_b64="", toploc_proof="",
                latency_ms=0.0, error=str(e),
            )

    def _find_backup(self, model_id: str, shard_range: tuple[int, int], exclude_node: str) -> Optional[str]:
        """Find an alternative node for a shard, excluding the failed one."""
        candidates = self._peers.with_shard(model_id, shard_range[0])
        for entry in candidates:
            if entry.record.node_id != exclude_node and entry.reputation >= 200:
                return entry.record.node_id
        return None
