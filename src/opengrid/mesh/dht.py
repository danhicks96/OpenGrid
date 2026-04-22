"""
Kademlia DHT wrapper for peer discovery.
Wraps kademlia (Python) or py-libp2p when available; falls back to a
simple in-process stub for single-machine dev/testing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Optional

log = logging.getLogger(__name__)

BOOTSTRAP_NODES: list[tuple[str, int]] = [
    # Hardcoded bootstrap nodes — replace with real IPs before public launch
    ("bootstrap1.opengrid.network", 7600),
    ("bootstrap2.opengrid.network", 7600),
]


@dataclass
class PeerRecord:
    node_id: str
    host: str
    port: int
    tier: str
    shards_hosted: list[str]
    last_seen: float = 0.0

    def key(self) -> bytes:
        return hashlib.sha1(self.node_id.encode()).digest()


class _InProcessDHT:
    """Minimal in-memory DHT used when kademlia library is not installed."""

    def __init__(self) -> None:
        self._table: dict[str, PeerRecord] = {}

    async def bootstrap(self, bootstrap_nodes: list[tuple[str, int]]) -> None:
        log.warning("Using in-process stub DHT — no real peer discovery.")

    async def set(self, key: str, value: str) -> None:
        pass  # stored locally only

    async def get(self, key: str) -> Optional[str]:
        return None

    def add_peer(self, record: PeerRecord) -> None:
        self._table[record.node_id] = record

    def get_peer(self, node_id: str) -> Optional[PeerRecord]:
        return self._table.get(node_id)

    def all_peers(self) -> list[PeerRecord]:
        return list(self._table.values())

    def peers_with_shard(self, model_shard_key: str) -> list[PeerRecord]:
        return [p for p in self._table.values() if model_shard_key in p.shards_hosted]


class DHTNode:
    """
    Public interface for DHT operations.
    Tries to use the `kademlia` package; falls back to the in-process stub.
    """

    def __init__(self, node_id: str, host: str = "0.0.0.0", port: int = 7600):
        self.node_id = node_id
        self.host = host
        self.port = port
        self._impl: _InProcessDHT = _InProcessDHT()
        self._kademlia_server = None

    async def start(self, bootstrap: list[tuple[str, int]] | None = None) -> None:
        try:
            from kademlia.network import Server  # type: ignore
            self._kademlia_server = Server()
            await self._kademlia_server.listen(self.port)
            if bootstrap:
                await self._kademlia_server.bootstrap(bootstrap)
            log.info("DHT started on port %d (kademlia)", self.port)
        except ImportError:
            log.warning("kademlia not installed — using stub DHT")
            await self._impl.bootstrap(bootstrap or BOOTSTRAP_NODES)

    async def announce(self, record: PeerRecord) -> None:
        """Publish this node's presence to the DHT."""
        self._impl.add_peer(record)
        if self._kademlia_server:
            await self._kademlia_server.set(
                record.node_id, json.dumps(asdict(record))
            )

    async def lookup(self, node_id: str) -> Optional[PeerRecord]:
        """Look up a peer by node_id."""
        local = self._impl.get_peer(node_id)
        if local:
            return local
        if self._kademlia_server:
            raw = await self._kademlia_server.get(node_id)
            if raw:
                return PeerRecord(**json.loads(raw))
        return None

    def peers_for_shard(self, model_id: str, shard_id: int) -> list[PeerRecord]:
        key = f"{model_id}:{shard_id}"
        return self._impl.peers_with_shard(key)

    def all_peers(self) -> list[PeerRecord]:
        return self._impl.all_peers()

    def register_peer(self, record: PeerRecord) -> None:
        """Directly register a peer (used when receiving gossip messages)."""
        record.last_seen = time.time()
        self._impl.add_peer(record)

    async def stop(self) -> None:
        if self._kademlia_server:
            self._kademlia_server.stop()
