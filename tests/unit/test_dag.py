"""Unit tests for the DAG executor."""
import pytest
from opengrid.coordinator.dag_executor import build_pipeline_dag, TaskStatus


def test_pipeline_dag_built():
    assignments = [
        ((0, 7), "node-a"),
        ((8, 15), "node-b"),
        ((16, 23), "node-c"),
    ]
    dag = build_pipeline_dag("job-1", "llama3-8b-int4", assignments)
    assert len(dag.tasks) == 3
    # First task has no deps
    first = dag.tasks[0]
    assert dag.dependencies[first.task_id] == []
    # Second task depends on first
    second = dag.tasks[1]
    assert dag.dependencies[second.task_id] == [first.task_id]


def test_ready_tasks_linear():
    assignments = [((0, 7), "a"), ((8, 15), "b")]
    dag = build_pipeline_dag("job-2", "m", assignments)
    ready = dag.ready_tasks()
    assert len(ready) == 1
    assert ready[0].node_id == "a"

    dag.mark_done(ready[0].task_id, result=b"act")
    ready2 = dag.ready_tasks()
    assert len(ready2) == 1
    assert ready2[0].node_id == "b"


def test_is_complete():
    assignments = [((0, 7), "a")]
    dag = build_pipeline_dag("job-3", "m", assignments)
    assert not dag.is_complete()
    dag.mark_done(dag.tasks[0].task_id)
    assert dag.is_complete()
