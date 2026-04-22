"""Unit tests for WorkUnitMonitor fault tolerance."""
import asyncio
import pytest
from unittest.mock import MagicMock

from opengrid.coordinator.dag_executor import build_pipeline_dag, TaskStatus
from opengrid.orchestrator.work_monitor import WorkUnitMonitor


def _make_monitor():
    peer_registry = MagicMock()
    peer_registry.with_shard.return_value = []
    scheduler = MagicMock()
    return WorkUnitMonitor(peer_registry, scheduler)


def test_register_and_status():
    monitor = _make_monitor()
    dag = build_pipeline_dag("job-1", "llama3-8b-int4", [((0, 7), "nodeA"), ((8, 15), "nodeB")])
    monitor.register_job(dag)
    status = monitor.status("job-1")
    assert status["job_id"] == "job-1"
    assert len(status["tasks"]) == 2


def test_checkpoint_and_done():
    monitor = _make_monitor()
    dag = build_pipeline_dag("job-2", "m", [((0, 7), "nodeA")])
    monitor.register_job(dag)
    task_id = dag.tasks[0].task_id
    dag.mark_running(task_id)
    monitor.mark_task_done("job-2", task_id, b"activations")
    # Job should be cleaned up
    assert monitor.status("job-2").get("error") is not None


def test_reassign_no_running_tasks():
    monitor = _make_monitor()
    dag = build_pipeline_dag("job-3", "m", [((0, 7), "nodeA")])
    monitor.register_job(dag)
    result = monitor.reassign("job-3", "nodeA")
    # No tasks are RUNNING yet, so nothing to reassign
    assert result["reassigned"] == 0


def test_active_job_count():
    monitor = _make_monitor()
    dag = build_pipeline_dag("job-4", "m", [((0, 7), "nodeA")])
    monitor.register_job(dag)
    assert monitor.active_job_count() == 1
