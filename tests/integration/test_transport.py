"""Integration tests for the transport layer — worker server, gossip listener, DAG dispatcher."""
import asyncio
import base64
import json
import pytest
import websockets
from unittest.mock import MagicMock

from opengrid.node.server import WorkerServer
from opengrid.node.worker import Worker
from opengrid.node.kv_cache import KVCacheStore
from opengrid.coordinator.dag_executor import build_pipeline_dag
from opengrid.coordinator.executor import DAGDispatcher
from opengrid.mesh.gossip import _validate_health
from opengrid.mesh.peer_registry import PeerRegistry


# ── Gossip validation tests ──────────────────────────────────────────────

def test_gossip_rejects_absurd_vram():
    assert not _validate_health({"vram_free_gb": 999, "ram_free_gb": 8, "jobs_active": 0, "avg_latency_ms": 10, "shards_hosted": []})


def test_gossip_rejects_negative_vram():
    assert not _validate_health({"vram_free_gb": -1, "ram_free_gb": 8, "jobs_active": 0, "avg_latency_ms": 10, "shards_hosted": []})


def test_gossip_rejects_absurd_jobs():
    assert not _validate_health({"vram_free_gb": 8, "ram_free_gb": 8, "jobs_active": 9999, "avg_latency_ms": 10, "shards_hosted": []})


def test_gossip_accepts_sane_values():
    assert _validate_health({"vram_free_gb": 24.0, "ram_free_gb": 64.0, "jobs_active": 5, "avg_latency_ms": 150, "shards_hosted": ["llama3-8b-int4:0"]})


# ── Worker fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_worker_no_model():
    """Worker with no model loaded — returns no_model_loaded error."""
    backend = MagicMock()
    backend.is_loaded.return_value = False
    backend.forward.side_effect = NotImplementedError
    kv = KVCacheStore(max_ram_gb=0.1)
    shard_mgr = MagicMock()
    return Worker("test-node", shard_mgr, backend, kv)


@pytest.fixture
def mock_worker_with_model():
    """Worker with a mock model that returns a fixed response."""
    backend = MagicMock()
    backend.is_loaded.return_value = True
    backend.generate.return_value = "Hello from the mock model!"
    kv = KVCacheStore(max_ram_gb=0.1)
    shard_mgr = MagicMock()
    return Worker("test-node", shard_mgr, backend, kv)


# ── Worker server tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_worker_server_starts_and_stops(mock_worker_no_model):
    server = WorkerServer(mock_worker_no_model, host="127.0.0.1", port=17600)
    await server.start()
    assert server.connection_count == 0
    await server.stop()


@pytest.mark.asyncio
async def test_worker_server_no_model_returns_error(mock_worker_no_model):
    """Without a model loaded, worker returns a clear error."""
    server = WorkerServer(mock_worker_no_model, host="127.0.0.1", port=17601)
    await server.start()
    try:
        async with websockets.connect("ws://127.0.0.1:17601") as ws:
            packet = {
                "packet_type": "inference_forward",
                "job_id": "test-job-1",
                "sequence_id": 1,
                "model_id": "test-model",
                "shard_range": [0, 7],
                "input_activations_b64": base64.b64encode(b"hello").decode(),
                "kv_cache_token": "test",
                "return_address": "",
                "deadline_ms": 5000,
            }
            await ws.send(json.dumps(packet))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)
            assert data["job_id"] == "test-job-1"
            assert data["error"] == "no_model_loaded"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_worker_server_handles_packet(mock_worker_with_model):
    """With a model loaded, worker generates text and returns it."""
    server = WorkerServer(mock_worker_with_model, host="127.0.0.1", port=17602)
    await server.start()
    try:
        async with websockets.connect("ws://127.0.0.1:17602") as ws:
            packet = {
                "packet_type": "inference_generate",
                "job_id": "test-job-2",
                "sequence_id": 1,
                "model_id": "test-model",
                "shard_range": [0, 7],
                "input_activations_b64": base64.b64encode(b"Hello, who are you?").decode(),
                "kv_cache_token": "test",
                "return_address": "",
                "deadline_ms": 5000,
            }
            await ws.send(json.dumps(packet))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(response)
            assert data["job_id"] == "test-job-2"
            assert data["error"] == ""
            assert data["latency_ms"] > 0
            assert data["tokens_generated"] > 0
            # Decode the output — should be our mock response
            output = base64.b64decode(data["output_activations_b64"]).decode()
            assert "Hello from the mock model" in output
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_worker_server_rejects_replay(mock_worker_with_model):
    """Second send of the same job_id+sequence_id gets rejected."""
    server = WorkerServer(mock_worker_with_model, host="127.0.0.1", port=17603)
    await server.start()
    try:
        packet = {
            "packet_type": "inference_generate",
            "job_id": "replay-job",
            "sequence_id": 1,
            "model_id": "test",
            "shard_range": [0, 7],
            "input_activations_b64": base64.b64encode(b"test prompt").decode(),
            "kv_cache_token": "",
            "return_address": "",
            "deadline_ms": 5000,
        }
        async with websockets.connect("ws://127.0.0.1:17603") as ws:
            # First send: should succeed
            await ws.send(json.dumps(packet))
            r1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert r1["error"] == ""

            # Second send (same job_id + sequence_id): should be rejected
            await ws.send(json.dumps(packet))
            r2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert r2["error"] == "replay_rejected"
    finally:
        await server.stop()


# ── DAG dispatcher tests ─────────────────────────────────────────────────

def test_dag_dispatcher_creates():
    registry = PeerRegistry()
    dispatcher = DAGDispatcher(registry)
    assert dispatcher is not None


@pytest.mark.asyncio
async def test_dag_dispatcher_fails_gracefully_no_peers():
    registry = PeerRegistry()
    dispatcher = DAGDispatcher(registry)
    dag = build_pipeline_dag("job-1", "model", [((0, 7), "nonexistent-node")])
    result = await dispatcher.execute(dag, input_text="test")
    assert not result.success
    assert "not found" in result.error
