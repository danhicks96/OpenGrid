"""
WorkUnitMonitor — the dNet-style fault tolerance layer.

Watches all active jobs. When a node goes dark mid-job:
  1. Detects the dropout via missed heartbeat deadline
  2. Finds the last good activation checkpoint for that shard
  3. Reassigns the shard to a backup node
  4. Resumes generation from the checkpoint — NOT from the beginning

This is the core reliability primitive. Without it, any node dropout
kills the entire generation. With it, users see a brief stutter then
output resumes.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Optional

from opengrid.coordinator.dag_executor import InferenceDAG, TaskStatus

log = logging.getLogger(__name__)

# How long (ms) to wait past a task's deadline before declaring it failed
DROPOUT_GRACE_MS = 500
MONITOR_INTERVAL = 1.0  # seconds between watchdog ticks


@dataclass
class JobRecord:
    dag: InferenceDAG
    started_at: float = field(default_factory=time.time)
    # task_id → wall-clock deadline (epoch seconds)
    deadlines: dict[str, float] = field(default_factory=dict)
    # task_id → last saved activation bytes (checkpoint)
    checkpoints: dict[str, bytes] = field(default_factory=dict)
    # task_id → node_id (current assignment, may change on reassignment)
    assignments: dict[str, str] = field(default_factory=dict)


class WorkUnitMonitor:
    """
    Background watchdog that detects node dropouts and triggers reassignment.
    Plugs into the coordinator; the orchestrator model can also call reassign()
    directly via its tool interface.
    """

    def __init__(self, peer_registry, scheduler):
        self._peers = peer_registry
        self._scheduler = scheduler
        self._jobs: dict[str, JobRecord] = {}
        self._lock = RLock()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._watchdog_loop())
        log.info("WorkUnitMonitor watchdog started.")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    def register_job(self, dag: InferenceDAG) -> None:
        record = JobRecord(dag=dag)
        for task in dag.tasks:
            record.assignments[task.task_id] = task.node_id
        with self._lock:
            self._jobs[dag.job_id] = record
        log.debug("Registered job %s (%d tasks)", dag.job_id, len(dag.tasks))

    def checkpoint(self, job_id: str, task_id: str, activations: bytes) -> None:
        """Save activation output from a completed shard — used for recovery."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record:
                record.checkpoints[task_id] = activations

    def set_deadline(self, job_id: str, task_id: str, deadline_ms: int) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record:
                record.deadlines[task_id] = time.time() + deadline_ms / 1000.0

    def mark_task_done(self, job_id: str, task_id: str, activations: bytes) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record:
                record.dag.mark_done(task_id, result=activations)
                record.checkpoints[task_id] = activations
                record.deadlines.pop(task_id, None)

        # Clean up completed jobs
        self._maybe_close(job_id)

    def _maybe_close(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record and record.dag.is_complete():
                del self._jobs[job_id]
                log.debug("Job %s complete — removed from monitor.", job_id)

    async def _watchdog_loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                log.exception("WorkUnitMonitor tick error")
            await asyncio.sleep(MONITOR_INTERVAL)

    def _tick(self) -> None:
        now = time.time()
        with self._lock:
            jobs = list(self._jobs.items())

        for job_id, record in jobs:
            for task in record.dag.tasks:
                if record.dag.status.get(task.task_id) != TaskStatus.RUNNING:
                    continue
                deadline = record.deadlines.get(task.task_id)
                if deadline and now > deadline + DROPOUT_GRACE_MS / 1000.0:
                    node_id = record.assignments[task.task_id]
                    log.warning(
                        "Node %s missed deadline for job %s task %s — triggering reassignment",
                        node_id, job_id, task.task_id,
                    )
                    asyncio.create_task(self._reassign_task(job_id, task.task_id, node_id))

    async def _reassign_task(self, job_id: str, task_id: str, failed_node_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            task = next((t for t in record.dag.tasks if t.task_id == task_id), None)
            if task is None or record.dag.status.get(task_id) != TaskStatus.RUNNING:
                return

        # Find a backup node for this shard
        model_id = task.model_id
        shard_lo, shard_hi = task.shard_range
        candidates = self._peers.with_shard(model_id, shard_lo)
        backup = next(
            (e.record.node_id for e in candidates if e.record.node_id != failed_node_id),
            None,
        )

        if backup is None:
            log.error("No backup node for %s shard %s-%s — job %s will stall",
                      model_id, shard_lo, shard_hi, job_id)
            with self._lock:
                record = self._jobs.get(job_id)
                if record:
                    record.dag.mark_failed(task_id, error="no backup node available")
            return

        log.info("Reassigning task %s from %s → %s (job %s)",
                 task_id, failed_node_id, backup, job_id)

        with self._lock:
            record = self._jobs.get(job_id)
            if record:
                task.node_id = backup
                task.retry_count += 1
                record.assignments[task_id] = backup
                record.dag.status[task_id] = TaskStatus.PENDING
                task.status = TaskStatus.PENDING
                # Restore deadline
                record.deadlines[task_id] = time.time() + task.deadline_ms / 1000.0

        # The DAG executor will pick up the now-PENDING task on its next dispatch cycle.
        # Activation input is recovered from the upstream checkpoint.

    def reassign(self, job_id: str, failed_node_id: str) -> dict:
        """Public interface for the orchestrator tool to call directly."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return {"error": f"Job {job_id} not found"}
            affected = [
                t.task_id for t in record.dag.tasks
                if record.assignments.get(t.task_id) == failed_node_id
                   and record.dag.status.get(t.task_id) == TaskStatus.RUNNING
            ]

        if not affected:
            return {"reassigned": 0, "message": "No running tasks on that node"}

        for task_id in affected:
            asyncio.create_task(self._reassign_task(job_id, task_id, failed_node_id))

        return {"reassigned": len(affected), "task_ids": affected}

    def status(self, job_id: str) -> dict:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is None:
            return {"error": f"Job {job_id} not found or already complete"}
        return {
            "job_id": job_id,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "node_id": record.assignments.get(t.task_id, t.node_id),
                    "shard_range": list(t.shard_range),
                    "status": record.dag.status.get(t.task_id, TaskStatus.PENDING).value,
                    "retries": t.retry_count,
                }
                for t in record.dag.tasks
            ],
        }

    def active_job_count(self) -> int:
        with self._lock:
            return len(self._jobs)
