"""
Gossip protocol — broadcasts node health/load metrics to sqrt(N) peers every 5 s.
Uses anti-entropy: nodes exchange summaries and reconcile stale state.

v0.0.3: Added OPENGRID_GOSSIP_SEEDS support for direct TCP bootstrap,
         bypassing UDP DHT entirely. Fixes NAT/UDP-blocked networks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

from opengrid.mesh.dht import DHTNode, PeerRecord

log = logging.getLogger(__name__)

GOSSIP_INTERVAL = 5.0   # seconds between gossip rounds
PEER_DEAD_AFTER = 30.0  # seconds without a heartbeat → remove from active set
GOSSIP_PORT_OFFSET = 1   # gossip listens on worker_port + 1
MAX_GOSSIP_MSG_SIZE = 8192

# Sane bounds for gossip validation
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
    # New: advertise the worker WebSocket port so peers know where to send work
    worker_port: int = 7610

    def is_stale(self) -> bool:
        return (time.time() - self.timestamp) > PEER_DEAD_AFTER


def _validate_health(data: dict) -> bool:
    """Reject gossip messages with absurd values."""
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


def _parse_seeds(env_var: str = "OPENGRID_GOSSIP_SEEDS") -> list[tuple[str, int]]:
    """
    Parse OPENGRID_GOSSIP_SEEDS env var.
    Format: "host1:port1,host2:port2"
    These are direct TCP gossip endpoints — bypass DHT entirely.
    """
    raw = os.environ.get(env_var, "")
    if not raw:
        return []
    seeds = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        host, _, port = entry.partition(":")
        seeds.append((host, int(port) if port else 7601))
    return seeds


class GossipNode:
    def __init__(self, dht: DHTNode, local_health_fn, gossip_port: int = 0):
        self.dht = dht
        self._local_health_fn = local_health_fn
        self._gossip_port = gossip_port or (dht.port + GOSSIP_PORT_OFFSET)
        self._seen: dict[str, NodeHealth] = {}
        self._seq = 0
        self._running = False
        self._send_task: Optional[asyncio.Task] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._server: Optional[asyncio.Server] = None

        # Direct TCP seed peers — used when UDP DHT is blocked by NAT
        self._tcp_seeds: list[tuple[str, int]] = _parse_seeds()

        # Rate limiting
        self._msg_counts: dict[str, int] = {}
        self._msg_limit_per_interval = 5

    async def start(self) -> None:
        self._running = True
        self._send_task = asyncio.create_task(self._send_loop())
        self._listen_task = asyncio.create_task(self._start_listener())
        if self._tcp_seeds:
            log.info("Gossip TCP seeds: %s", self._tcp_seeds)
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
        remote = writer.get_extra_info("peername")
        try:
            # Read all available data (handles messages with or without newlines)
            chunks = []
            try:
                while True:
                    chunk = await asyncio.wait_for(reader.read(MAX_GOSSIP_MSG_SIZE), timeout=2.0)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk or len(b"".join(chunks)) > MAX_GOSSIP_MSG_SIZE:
                        break
            except asyncio.TimeoutError:
                pass
            
            data = b"".join(chunks)
            if not data:
                return
            line = data.decode(errors="replace").strip()

            peer_ip = remote[0] if remote else "unknown"
            count = self._msg_counts.get(peer_ip, 0)
            if count >= self._msg_limit_per_interval:
                log.debug("Rate limiting gossip from %s", peer_ip)
                return
            self._msg_counts[peer_ip] = count + 1

            # Accept both "GOSSIP {...}" and raw "{...}" formats
            if line.startswith("GOSSIP "):
                raw = line[7:]
            elif line.startswith("{"):
                raw = line
            else:
                log.debug("Unknown gossip format from %s: %s", peer_ip, line[:80])
                return
            self.receive(raw, peer_host=peer_ip, peer_port=remote[1] if remote else 0)
        except Exception as e:
            log.debug("Gossip receive error from %s: %s", remote, e)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    # ── Sender ────────────────────────────────────────────────────────────

    async def _send_loop(self) -> None:
        while self._running:
            try:
                await self._gossip_round()
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

        msg = json.dumps(asdict(health))

        # 1) Gossip to DHT-discovered peers (random subset)
        peers = self.dht.all_peers()
        fanout = max(1, int(math.sqrt(len(peers))))
        targets = random.sample(peers, min(fanout, len(peers))) if peers else []
        for peer in targets:
            gossip_port = peer.port + GOSSIP_PORT_OFFSET if peer.port else self._gossip_port
            await self._send_tcp(peer.host, gossip_port, msg)

        # 2) ALWAYS gossip to direct TCP seeds — these bypass DHT/UDP entirely
        for host, port in self._tcp_seeds:
            await self._send_tcp(host, port, msg)

        # Prune dead peers
        dead = [nid for nid, h in self._seen.items() if h.is_stale()]
        for nid in dead:
            log.debug("Pruning stale peer %s", nid)
            del self._seen[nid]

    async def _send_tcp(self, host: str, port: int, payload: str) -> None:
        """Send a gossip message to a peer over TCP. Fire-and-forget."""
        if not host or not port:
            return
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=2.0
            )
            writer.write(b"GOSSIP " + payload.encode() + b"\n")
            await writer.drain()
            writer.close()
        except Exception as e:
            log.debug("Gossip send to %s:%d failed: %s", host, port, e)

    # ── Receive / state update ────────────────────────────────────────────

    def receive(self, raw: str, peer_host: str = "", peer_port: int = 0) -> None:
        if len(raw) > MAX_GOSSIP_MSG_SIZE:
            log.warning("Gossip message too large (%d bytes), dropping", len(raw))
            return
        try:
            # Try to parse as-is first
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Attempt to repair truncated JSON — close any open brackets/braces
                repaired = raw.rstrip()
                if not repaired.endswith("}"):
                    # Truncated mid-value: find last complete key-value pair
                    last_comma = repaired.rfind(",")
                    if last_comma > 0:
                        repaired = repaired[:last_comma] + "}"
                    else:
                        repaired = repaired + "}"
                data = json.loads(repaired)
                log.debug("Repaired truncated gossip from %s", peer_host)
            if not _validate_health(data):
                log.warning("Gossip validation failed from %s: absurd values", peer_host)
                return

            # Handle NodeHealth with or without worker_port field (backwards compat)
            if "worker_port" not in data:
                data["worker_port"] = 7610

            health = NodeHealth(**data)
            existing = self._seen.get(health.node_id)
            if existing is None or health.seq > existing.seq:
                self._seen[health.node_id] = health
                self.dht.register_peer(PeerRecord(
                    node_id=health.node_id,
                    host=peer_host or "",
                    port=health.worker_port,  # store the WORKER port, not gossip port
                    tier=health.tier,
                    shards_hosted=health.shards_hosted,
                    last_seen=health.timestamp,
                ))
                if existing is None:
                    log.info("New peer discovered via gossip: %s (%s) at %s:%d",
                             health.node_id, health.tier, peer_host, health.worker_port)
        except Exception:
            log.warning("Malformed gossip message: %r", raw[:200])

    def active_peers(self) -> list[NodeHealth]:
        return [h for h in self._seen.values() if not h.is_stale()]

    def peer_health(self, node_id: str) -> Optional[NodeHealth]:
        return self._seen.get(node_id)
