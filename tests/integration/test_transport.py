"""Integration tests for the transport layer — worker server, gossip listener, DAG dispatcher."""
import asyncio
import base64
import json
import pytest
import websockets
from unittest.mock import MagicMock, AsyncMock

from opengrid.node.server import WorkerServer
from opengrid.node.worker import Worker, WorkPacket, WorkResult
from opengrid.node.kv_cache import KVCacheStore
from opengrid.node.inference_engine import LlamaCppBackend
from opengrid.coordinator.dag_executor import build_pipeline_dag
from opengrid.coordinator.executor import DAGDispatcher
from opengrid.mesh.gossip import GossipNode, NodeHealth, _validate_health
from opengrid.mesh.peer_registry import PeerRegistry
from opengrid.mesh.dht import DHTNode, PeerRecord
from opengrid.daemon.shard_manager import ShardManager


# ── Gossip validation tests ──────────────────────────────────────────────

def test_gossip_rejects_absurd_vram():
    assert not _validate_health({"vram_free_gb": 999, "ram_free_gb": 8, "jobs_active": 0, "avg_latency_ms": 10, "shards_hosted": []})


def test_gossip_rejects_negative_vram():
    assert not _validate_health({"vram_free_gb": -1, "ram_free_gb": 8, "jobs_active": 0, "avg_latency_ms": 10, "shards_hosted": []})


def test_gossip_rejects_absurd_jobs():
    assert not _validate_health({"vram_free_gb": 8, "ram_free_gb": 8, "jobs_active": 9999, "avg_latency_ms": 10, "shards_hosted": []})


def test_gossip_accepts_sane_values():
    assert _validate_health({"vram_free_gb": 24.0, "ram_free_gb": 64.0, "jobs_active": 5, "avg_latency_ms": 150, "shards_hosted": ["llama3-8b-int4:0"]})


# ── Worker server tests ──────────────────────────────────────────────────

@pytest.fixture
def mock_worker():
    backend = MagicMock()
    backend.forward.side_effect = NotImplementedError
    kv = KVCacheStore(max_ram_gb=0.1)
    shard_mgr = MagicMock()
    return Worker("test-node", shard_mgr, backend, kv)


@pytest.mark.asyncio
async def test_worker_server_starts_and_stops(mock_worker):
    server = WorkerServer(mock_worker, host="127.0.0.1", port=17600)
    await server.start()
    assert server.connection_count == 0
    await server.stop()


@pytest.mark.asyncio
async def test_worker_server_handles_packet(mock_worker):
    server = WorkerServer(mock_worker, host="127.0.0.1", port=17601)
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
            assert data["error"] == ""
            assert data["latency_ms"] > 0
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_worker_server_rejects_replay(mock_worker):
    server = WorkerServer(mock_worker, host="127.0.0.1", port=17602)
    await server.start()
    try:
        packet = {
            "packet_type": "inference_forward",
            "job_id": "replay-job",
            "sequence_id": 1,
            "model_id": "test",
            "shard_range": [0, 7],
            "input_activations_b64": base64.b64encode(b"test").decode(),
            "kv_cache_token": "",
            "return_address": "",
            "deadline_ms": 5000,
        }
        async with websockets.connect("ws://127.0.0.1:17602") as ws:
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
