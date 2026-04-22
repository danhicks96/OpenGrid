"""
FastAPI application — wires all components together.

v0.0.2: Added DAG dispatcher, worker server, signal handlers for graceful shutdown.

Modes:
  standard    — coordinator + API only (no local model)
  orchestrator — coordinator + API + local micro model tool-caller
  worker      — worker node only (no API, just WebSocket server for work packets)
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from opengrid.api.routes import completions, credits, models
from opengrid.coordinator.admission import AdmissionController
from opengrid.coordinator.kv_router import KVRouter
from opengrid.coordinator.scheduler import Scheduler
from opengrid.daemon.config import load_config, ensure_dirs
from opengrid.mesh.dht import DHTNode
from opengrid.mesh.gossip import GossipNode
from opengrid.mesh.peer_registry import PeerRegistry
from opengrid.registry.model_registry import ModelRegistry

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    ensure_dirs(cfg)
    mode = os.environ.get("OPENGRID_MODE", "standard")

    # ── Credit ledger ─────────────────────────────────────────────────────
    try:
        from opengrid.daemon.credit_ledger import CreditLedger
        app.state.ledger = CreditLedger(cfg.ledger_path)
    except Exception as e:
        log.warning("Credit ledger init failed: %s", e)
        app.state.ledger = None

    # ── Model registry ────────────────────────────────────────────────────
    app.state.model_registry = ModelRegistry()

    # ── Peer registry ─────────────────────────────────────────────────────
    peer_registry = PeerRegistry()
    app.state.peer_registry = peer_registry

    # ── Hardware profile ──────────────────────────────────────────────────
    from opengrid.daemon.benchmark import load_or_run
    profile = load_or_run(cfg.profile_path)

    # ── DHT ───────────────────────────────────────────────────────────────
    dht = DHTNode(node_id=profile.node_id, port=cfg.network.listen_port)
    bootstrap_env = os.environ.get("OPENGRID_BOOTSTRAP_PEERS", "")
    extra_bootstrap = []
    if bootstrap_env:
        for peer in bootstrap_env.split(","):
            host, _, port = peer.strip().partition(":")
            extra_bootstrap.append((host, int(port) if port else 7600))
    await dht.start(bootstrap=extra_bootstrap or None)
    app.state.dht = dht

    # ── Gossip ────────────────────────────────────────────────────────────
    def _local_health():
        from opengrid.mesh.gossip import NodeHealth
        import time
        worker_srv = getattr(app.state, "worker_server", None)
        jobs = worker_srv._worker.active_job_count() if worker_srv else 0
        return NodeHealth(
            node_id=profile.node_id,
            timestamp=time.time(),
            seq=0,
            status="active",
            vram_free_gb=profile.gpu_vram_free_gb,
            ram_free_gb=profile.ram_free_gb,
            jobs_active=jobs,
            avg_latency_ms=0.0,
            shards_hosted=[],
            tier=profile.tier,
        )

    gossip = GossipNode(dht, _local_health)
    await gossip.start()
    app.state.gossip = gossip

    # ── Coordinator stack ─────────────────────────────────────────────────
    kv_router = KVRouter(peer_registry)
    admission = AdmissionController(peer_registry, gossip)
    scheduler = Scheduler(peer_registry, app.state.model_registry, admission, kv_router)
    app.state.scheduler = scheduler

    # ── DAG dispatcher (new in v0.0.2) ────────────────────────────────────
    from opengrid.coordinator.reputation import ReputationManager
    reputation = ReputationManager(peer_registry)
    app.state.reputation = reputation

    from opengrid.coordinator.executor import DAGDispatcher
    dispatcher = DAGDispatcher(peer_registry, reputation)
    app.state.dispatcher = dispatcher

    # ── Work unit monitor (fault tolerance) ───────────────────────────────
    from opengrid.orchestrator.work_monitor import WorkUnitMonitor
    monitor = WorkUnitMonitor(peer_registry, scheduler)
    await monitor.start()
    app.state.work_monitor = monitor

    # ── Worker WebSocket server (new in v0.0.2) ───────────────────────────
    # Every node runs a worker server so it can serve inference jobs
    from opengrid.node.inference_engine import select_backend
    from opengrid.node.kv_cache import KVCacheStore
    from opengrid.node.worker import Worker
    from opengrid.node.server import WorkerServer
    from opengrid.daemon.shard_manager import ShardManager

    backend = select_backend(gpu_vram_gb=profile.gpu_vram_free_gb, platform=profile.platform)
    kv_store = KVCacheStore(max_ram_gb=cfg.resources.max_ram_gb)
    shard_mgr = ShardManager(cfg.shards_dir, cfg.resources.max_disk_gb)
    worker = Worker(profile.node_id, shard_mgr, backend, kv_store)

    # Auto-load model if OPENGRID_MODEL_PATH is set (from --model CLI flag)
    model_path = os.environ.get("OPENGRID_MODEL_PATH", "")
    if model_path:
        try:
            worker.load_model(model_path)
            log.info("Model loaded from CLI: %s", model_path)
        except Exception as e:
            log.warning("Failed to load model %s: %s", model_path, e)

    worker_port = cfg.network.listen_port + 10  # 7610: WebSocket worker (7600 is DHT)
    worker_server = WorkerServer(worker, host="0.0.0.0", port=worker_port)
    await worker_server.start()
    app.state.worker_server = worker_server

    # ── Orchestrator model (orchestrator mode only) ───────────────────────
    app.state.orchestrator = None
    if mode == "orchestrator":
        from opengrid.orchestrator.tools import OrchestratorTools
        from opengrid.orchestrator.local_model import LocalOrchestratorModel

        model_path_str = os.environ.get("OPENGRID_ORCHESTRATOR_MODEL", "")
        model_path = Path(model_path_str) if model_path_str else (
            cfg.home / "models" / "bitnet-b158-2b.gguf"
        )
        local_model = LocalOrchestratorModel(model_path=model_path)
        orch_tools = OrchestratorTools(scheduler, peer_registry, gossip, monitor, kv_router)
        app.state.orchestrator = (local_model, orch_tools)
        log.info(
            "Orchestrator mode: local model %s",
            "loaded" if local_model.is_loaded() else "rule-based fallback",
        )

    # ── Graceful shutdown handler (edge case #41) ─────────────────────────
    _shutdown_event = asyncio.Event()

    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        log.info("Received %s — initiating graceful shutdown...", sig_name)
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    log.info(
        "OpenGrid ready — port %d | node %s | tier %s | mode %s | "
        "worker ws://0.0.0.0:%d | gossip :%d | models: %s",
        cfg.network.api_port, profile.node_id, profile.tier, mode,
        cfg.network.listen_port, cfg.network.listen_port + 1, worker_port,
        ", ".join(app.state.model_registry.list_models()) or "(none loaded)",
    )

    yield

    # ── Shutdown sequence ─────────────────────────────────────────────────
    log.info("Shutting down...")
    await worker_server.stop()
    await monitor.stop()
    await gossip.stop()
    await dht.stop()
    if app.state.ledger:
        app.state.ledger.close()
    log.info("Shutdown complete.")


app = FastAPI(
    title="OpenGrid",
    description="Distributed peer-to-peer LLM inference network",
    version="0.0.2a0",
    lifespan=lifespan,
)

app.include_router(completions.router)
app.include_router(models.router)
app.include_router(credits.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
