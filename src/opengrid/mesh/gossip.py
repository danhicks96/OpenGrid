"""
Gossip protocol — broadcasts node health/load metrics to sqrt(N) peers every 5 s.
Uses anti-entropy: nodes exchange summaries and reconcile stale state.

v0.0.2: Added TCP listener, random fanout, gossip value validation,
         peer host/port tracking from incoming connections.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

from opengrid.mesh.dht import DHTNode, PeerRecord

log = logging.getLogger(__name__)

GOSSIP_INTERVAL = 5.0   # seconds between gossip rounds
PEER_DEAD_AFTER = 30.0  # seconds without a heartbeat → remove from active set
GOSSIP_PORT_OFFSET = 1   # gossip listens on worker_port + 1
MAX_GOSSIP_MSG_SIZE = 8192  # reject messages larger than this

# Sane bounds for gossip validation (edge case #44: reject absurd claims)
MAX_VRAM_GB = 256.0
MAX_RAM_GB = 2048.0
MAX_JOBS_ACTIVE = 1000


@dataclass
class NodeHealth:
    node_id: str
    timestamp: float
    seq: int
    status: str          # "active" | "paused" | "overloaded" | "draining"
    vram_free_gb: float
    ram_free_gb: float
    jobs_active: int
    avg_latency_ms: float
    shards_hosted: list[str]
    tier: str

    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > PEER_DEAD_AFTER


def _validate_health(data: dict) -> bool:
    """Reject gossip messages with absurd values (edge case #44)."""
    if data.get("vram_free_gb", 0) > MAX_VRAM_GB:
        return False
    if data.get("ram_free_gb", 0) > MAX_RAM_GB:
        return False
    if data.get("jobs_active", 0) > MAX_JOBS_ACTIVE:
        return False
    if data.get("vram_free_gb", 0) < 0:
        return False
    if data.get("avg_latency_ms", 0) < 0:
        return False
    if not isinstance(data.get("shards_hosted"), list):
        return False
    return True


class GossipNode:
    def __init__(self, dht: DHTNode, local_health_fn, gossip_port: int = 0):
        """
        dht             — DHTNode used for peer enumeration
        local_health_fn — callable () → NodeHealth with current local state
        gossip_port     — port to listen for incoming gossip (0 = auto from DHT port)
        """
        self.dht = dht
        self._local_health_fn = local_health_fn
        self._gossip_port = gossip_port or (dht.port + GOSSIP_PORT_OFFSET)
        self._seen: dict[str, NodeHealth] = {}  # node_id → latest health
        self._seq = 0
        self._running = False
        self._send_task: Optional[asyncio.Task] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._server: Optional[asyncio.Server] = None

        # Rate limiting: track messages per peer per interval
        self._msg_counts: dict[str, int] = {}
        self._msg_limit_per_interval = 5  # max messages per peer per gossip interval

    async def start(self) -> None:
        self._running = True
        self._send_task = asyncio.create_task(self._send_loop())
        self._listen_task = asyncio.create_task(self._start_listener())
        log.info("Gossip started (send loop + listener on port %d).", self._gossip_port)

    async def stop(self) -> None:
        self._running = False
        if self._send_task:
            self._send_task.cancel()
        if self._listen_task:
            self._listen_task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # ── Listener ──────────────────────────────────────────────────────────

    async def _start_listener(self) -> None:
        """TCP server that receives gossip messages from peers."""
        try:
            self._server = await asyncio.start_server(
                self._handle_incoming, "0.0.0.0", self._gossip_port,
            )
            log.info("Gossip listener bound to port %d", self._gossip_port)
            async with self._server:
                await self._server.serve_forever()
        except OSError as e:
            log.warning("Gossip listener failed to bind port %d: %s", self._gossip_port, e)
        except asyncio.CancelledError:
            pass

    async def _handle_incoming(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single incoming gossip connection."""
        remote = writer.get_extra_info("peername")
        try:
            data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not data:
                return
            line = data.decode().strip()

            # Rate limit per peer IP
            peer_ip = remote[0] if remote else "unknown"
            count = self._msg_counts.get(peer_ip, 0)
            if count >= self._msg_limit_per_interval:
                log.debug("Rate limiting gossip from %s", peer_ip)
                return
            self._msg_counts[peer_ip] = count + 1

            if line.startswith("GOSSIP "):
                raw = line[7:]
                self.receive(raw, peer_host=peer_ip, peer_port=remote[1] if remote else 0)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            log.debug("Gossip receive error from %s: %s", remote, e)
        finally:
            writer.close()

    # ── Sender ────────────────────────────────────────────────────────────

    async def _send_loop(self) -> None:
        while self._running:
            try:
                await self._gossip_round()
                # Reset rate limit counters each interval
                self._msg_counts.clear()
            except asyncio.CancelledError:
                break
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
        # Random sample instead of deterministic first-N (spec requirement)
        targets = random.sample(peers, min(fanout, len(peers))) if peers else []

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
        gossip_port = peer.port + GOSSIP_PORT_OFFSET if peer.port else self._gossip_port
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.host, gossip_port), timeout=2.0
            )
            writer.write(b"GOSSIP " + payload.encode() + b"\n")
            await writer.drain()
            writer.close()
        except Exception as e:
            log.debug("Gossip send to %s:%d failed: %s", peer.host, gossip_port, e)

    # ── Receive / state update ────────────────────────────────────────────

    def receive(self, raw: str, peer_host: str = "", peer_port: int = 0) -> None:
        """Process a received gossip message and update local state."""
        if len(raw) > MAX_GOSSIP_MSG_SIZE:
            log.warning("Gossip message too large (%d bytes), dropping", len(raw))
            return
        try:
            data = json.loads(raw)

            # Validate bounds (edge case #44)
            if not _validate_health(data):
                log.warning("Gossip validation failed from %s: absurd values", peer_host)
                return

            health = NodeHealth(**data)
            existing = self._seen.get(health.node_id)
            if existing is None or health.seq > existing.seq:
                self._seen[health.node_id] = health
                # Register peer with actual host/port from the connection
                self.dht.register_peer(PeerRecord(
                    node_id=health.node_id,
                    host=peer_host or "",
                    port=peer_port or 0,
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
