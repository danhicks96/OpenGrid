"""
DAG execution engine — manages the state machine for a single inference request.
Each request becomes a DAG of Tasks (one per shard stage).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    node_id: str
    model_id: str
    shard_range: tuple[int, int]
    deadline_ms: int = 2000
    retry_count: int = 0
    max_retries: int = 2
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class InferenceDAG:
    job_id: str
    model_id: str
    tasks: list[Task] = field(default_factory=list)
    # task_id → list of upstream task_ids (pipeline: each stage depends on prior)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    status: dict[str, TaskStatus] = field(default_factory=dict)
    # task_id → output activations (bytes blob or tensor handle)
    activations: dict[str, bytes] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_task(self, task: Task, depends_on: list[str] | None = None) -> None:
        self.tasks.append(task)
        self.status[task.task_id] = TaskStatus.PENDING
        self.dependencies[task.task_id] = depends_on or []

    def ready_tasks(self) -> list[Task]:
        """Tasks whose dependencies are all DONE and which are still PENDING."""
        ready = []
        for task in self.tasks:
            if self.status[task.task_id] != TaskStatus.PENDING:
                continue
            deps = self.dependencies[task.task_id]
            if all(self.status.get(d) == TaskStatus.DONE for d in deps):
                ready.append(task)
        return ready

    def mark_running(self, task_id: str) -> None:
        self.status[task_id] = TaskStatus.RUNNING
        for t in self.tasks:
            if t.task_id == task_id:
                t.status = TaskStatus.RUNNING
                t.started_at = time.time()
                break

    def mark_done(self, task_id: str, result: Any = None) -> None:
        self.status[task_id] = TaskStatus.DONE
        for t in self.tasks:
            if t.task_id == task_id:
                t.status = TaskStatus.DONE
                t.result = result
                t.finished_at = time.time()
                break
        if result is not None:
            self.activations[task_id] = result

    def mark_failed(self, task_id: str, error: str = "") -> None:
        self.status[task_id] = TaskStatus.FAILED
        for t in self.tasks:
            if t.task_id == task_id:
                t.status = TaskStatus.FAILED
                t.error = error
                t.finished_at = time.time()
                break

    def is_complete(self) -> bool:
        return all(s == TaskStatus.DONE for s in self.status.values())

    def has_failures(self) -> bool:
        return any(s == TaskStatus.FAILED for s in self.status.values())

    def upstream_activations(self, task_id: str) -> Optional[bytes]:
        deps = self.dependencies.get(task_id, [])
        if not deps:
            return None
        return self.activations.get(deps[-1])


def build_pipeline_dag(
    job_id: str,
    model_id: str,
    shard_assignments: list[tuple[tuple[int, int], str]],  # [(shard_range, node_id), ...]
    deadline_ms: int = 2000,
) -> InferenceDAG:
    """
    Build a linear pipeline DAG from a list of (shard_range, node_id) pairs.
    Each stage depends on the previous stage.
    """
    dag = InferenceDAG(job_id=job_id, model_id=model_id)
    prev_id: Optional[str] = None
    for shard_range, node_id in shard_assignments:
        task = Task(
            task_id=str(uuid.uuid4()),
            node_id=node_id,
            model_id=model_id,
            shard_range=shard_range,
            deadline_ms=deadline_ms,
        )
        dag.add_task(task, depends_on=[prev_id] if prev_id else [])
        prev_id = task.task_id
    return dag
