"""
FastAPI application — wires all components together.
Entry point: uvicorn opengrid.api.server:app
"""
from __future__ import annotations

import logging
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    ensure_dirs(cfg)

    # Ledger (optional — omit if no DB path configured)
    try:
        from opengrid.daemon.credit_ledger import CreditLedger
        app.state.ledger = CreditLedger(cfg.ledger_path)
    except Exception as e:
        log.warning("Credit ledger init failed: %s", e)
        app.state.ledger = None

    # Model registry
    app.state.model_registry = ModelRegistry()

    # Peer registry + DHT + gossip
    peer_registry = PeerRegistry()
    app.state.peer_registry = peer_registry

    from opengrid.daemon.benchmark import load_or_run
    profile = load_or_run(cfg.profile_path)

    dht = DHTNode(node_id=profile.node_id, port=cfg.network.listen_port)
    await dht.start()
    app.state.dht = dht

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

    log.info("OpenGrid API ready on port %d (node: %s, tier: %s)",
             cfg.network.api_port, profile.node_id, profile.tier)

    yield

    # Shutdown
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
