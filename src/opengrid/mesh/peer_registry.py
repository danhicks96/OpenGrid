"""
In-memory peer table — maintains the live view of the network topology.
Updated by gossip receives and DHT lookups.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Optional

from opengrid.mesh.dht import PeerRecord


@dataclass
class PeerEntry:
    record: PeerRecord
    reputation: int = 500    # 0–1000, starts neutral
    jobs_completed: int = 0
    jobs_failed: int = 0
    last_latency_ms: float = 0.0
    last_seen: float = field(default_factory=time.time)

    def is_alive(self, timeout: float = 30.0) -> bool:
        return (time.time() - self.last_seen) < timeout


class PeerRegistry:
    def __init__(self):
        self._peers: dict[str, PeerEntry] = {}
        self._lock = RLock()

    def upsert(self, record: PeerRecord, reputation: int | None = None) -> None:
        with self._lock:
            existing = self._peers.get(record.node_id)
            if existing:
                existing.record = record
                existing.last_seen = time.time()
                if reputation is not None:
                    existing.reputation = max(0, min(1000, reputation))
            else:
                self._peers[record.node_id] = PeerEntry(
                    record=record,
                    reputation=reputation if reputation is not None else 500,
                )

    def get(self, node_id: str) -> Optional[PeerEntry]:
        return self._peers.get(node_id)

    def all_active(self) -> list[PeerEntry]:
        with self._lock:
            return [e for e in self._peers.values() if e.is_alive()]

    def by_tier(self, tier: str) -> list[PeerEntry]:
        return [e for e in self.all_active() if e.record.tier == tier]

    def with_shard(self, model_id: str, shard_id: int) -> list[PeerEntry]:
        key = f"{model_id}:{shard_id}"
        return [
            e for e in self.all_active()
            if key in e.record.shards_hosted and e.reputation >= 200
        ]

    def adjust_reputation(self, node_id: str, delta: int) -> None:
        with self._lock:
            entry = self._peers.get(node_id)
            if entry:
                entry.reputation = max(0, min(1000, entry.reputation + delta))

    def record_job_done(self, node_id: str, latency_ms: float) -> None:
        with self._lock:
            entry = self._peers.get(node_id)
            if entry:
                entry.jobs_completed += 1
                entry.last_latency_ms = latency_ms
                self.adjust_reputation(node_id, +1)

    def record_job_failed(self, node_id: str, fault: bool = True) -> None:
        with self._lock:
            entry = self._peers.get(node_id)
            if entry:
                entry.jobs_failed += 1
                self.adjust_reputation(node_id, -20 if fault else -2)
