"""
Gossip protocol — broadcasts node health/load metrics to sqrt(N) peers every 5 s.
Uses anti-entropy: nodes exchange summaries and reconcile stale state.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Optional

from opengrid.mesh.dht import DHTNode, PeerRecord

log = logging.getLogger(__name__)

GOSSIP_INTERVAL = 5.0  # seconds
PEER_DEAD_AFTER = 30.0  # seconds without a heartbeat → remove from active set


@dataclass
class NodeHealth:
    node_id: str
    timestamp: float
    seq: int
    status: str          # "active" | "paused" | "overloaded"
    vram_free_gb: float
    ram_free_gb: float
    jobs_active: int
    avg_latency_ms: float
    shards_hosted: list[str]
    tier: str

    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > PEER_DEAD_AFTER


class GossipNode:
    def __init__(self, dht: DHTNode, local_health_fn):
        """
        dht             — DHTNode used for peer enumeration
        local_health_fn — callable () → NodeHealth with current local state
        """
        self.dht = dht
        self._local_health_fn = local_health_fn
        self._seen: dict[str, NodeHealth] = {}  # node_id → latest health
        self._seq = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("Gossip loop started.")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._gossip_round()
            except Exception:
                log.exception("Error in gossip round")
            await asyncio.sleep(GOSSIP_INTERVAL)

    async def _gossip_round(self) -> None:
        self._seq += 1
        health = self._local_health_fn()
        health.seq = self._seq
        health.timestamp = time.time()
        self._seen[health.node_id] = health

        peers = self.dht.all_peers()
        fanout = max(1, int(math.sqrt(len(peers))))
        targets = peers[:fanout]  # in a real impl: random sample

        msg = json.dumps(asdict(health))
        for peer in targets:
            await self._send(peer, msg)

        # Prune dead peers
        dead = [nid for nid, h in self._seen.items() if h.is_stale()]
        for nid in dead:
            log.debug("Pruning stale peer %s", nid)
            del self._seen[nid]

    async def _send(self, peer: PeerRecord, payload: str) -> None:
        """Send a gossip message to a peer over TCP. Fire-and-forget."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.host, peer.port), timeout=2.0
            )
            writer.write(b"GOSSIP " + payload.encode() + b"\n")
            await writer.drain()
            writer.close()
        except Exception as e:
            log.debug("Gossip send to %s:%d failed: %s", peer.host, peer.port, e)

    def receive(self, raw: str) -> None:
        """Called when a gossip message is received from a remote peer."""
        try:
            data = json.loads(raw)
            health = NodeHealth(**data)
            existing = self._seen.get(health.node_id)
            if existing is None or health.seq > existing.seq:
                self._seen[health.node_id] = health
                self.dht.register_peer(PeerRecord(
                    node_id=health.node_id,
                    host="",   # host filled in by transport layer
                    port=0,
                    tier=health.tier,
                    shards_hosted=health.shards_hosted,
                    last_seen=health.timestamp,
                ))
        except Exception:
            log.warning("Malformed gossip message: %r", raw[:200])

    def active_peers(self) -> list[NodeHealth]:
        return [h for h in self._seen.values() if not h.is_stale()]

    def peer_health(self, node_id: str) -> Optional[NodeHealth]:
        return self._seen.get(node_id)
