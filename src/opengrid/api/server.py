"""
FastAPI application — wires all components together.

Modes:
  standard    — coordinator + API only (no local model)
  orchestrator — coordinator + API + local micro model tool-caller
"""
from __future__ import annotations

import logging
import os
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
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    ensure_dirs(cfg)

    # Credit ledger
    try:
        from opengrid.daemon.credit_ledger import CreditLedger
        app.state.ledger = CreditLedger(cfg.ledger_path)
    except Exception as e:
        log.warning("Credit ledger init failed: %s", e)
        app.state.ledger = None

    # Model registry
    app.state.model_registry = ModelRegistry()

    # Peer registry
    peer_registry = PeerRegistry()
    app.state.peer_registry = peer_registry

    # Hardware profile
    from opengrid.daemon.benchmark import load_or_run
    profile = load_or_run(cfg.profile_path)

    # DHT
    dht = DHTNode(node_id=profile.node_id, port=cfg.network.listen_port)
    bootstrap_env = os.environ.get("OPENGRID_BOOTSTRAP_PEERS", "")
    extra_bootstrap = []
    if bootstrap_env:
        for peer in bootstrap_env.split(","):
            host, _, port = peer.strip().partition(":")
            extra_bootstrap.append((host, int(port) if port else 7600))
    await dht.start(bootstrap=extra_bootstrap or None)
    app.state.dht = dht

    # Gossip
    def _local_health():
        from opengrid.mesh.gossip import NodeHealth
        import time
        return NodeHealth(
            node_id=profile.node_id,
            timestamp=time.time(),
            seq=0,
            status="active",
            vram_free_gb=profile.gpu_vram_free_gb,
            ram_free_gb=profile.ram_free_gb,
            jobs_active=0,
            avg_latency_ms=0.0,
            shards_hosted=[],
            tier=profile.tier,
        )

    gossip = GossipNode(dht, _local_health)
    await gossip.start()
    app.state.gossip = gossip

    # Coordinator stack
    kv_router = KVRouter(peer_registry)
    admission = AdmissionController(peer_registry, gossip)
    scheduler = Scheduler(peer_registry, app.state.model_registry, admission, kv_router)
    app.state.scheduler = scheduler

    # Work unit monitor (fault tolerance)
    from opengrid.orchestrator.work_monitor import WorkUnitMonitor
    monitor = WorkUnitMonitor(peer_registry, scheduler)
    await monitor.start()
    app.state.work_monitor = monitor

    # Local orchestrator model (only in orchestrator mode)
    mode = os.environ.get("OPENGRID_MODE", "standard")
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

    log.info(
        "OpenGrid API ready — port %d | node %s | tier %s | mode %s",
        cfg.network.api_port, profile.node_id, profile.tier, mode,
    )

    yield

    # Shutdown
    await monitor.stop()
    await gossip.stop()
    await dht.stop()
    if app.state.ledger:
        app.state.ledger.close()


app = FastAPI(
    title="OpenGrid",
    description="Distributed peer-to-peer LLM inference network",
    version="0.0.1a0",
    lifespan=lifespan,
)

app.include_router(completions.router)
app.include_router(models.router)
app.include_router(credits.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
