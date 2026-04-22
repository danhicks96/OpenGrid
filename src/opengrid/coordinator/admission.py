"""
Admission control — checks node eligibility before assigning a task.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from opengrid.mesh.gossip import GossipNode
from opengrid.mesh.peer_registry import PeerRegistry


@dataclass
class AdmissionResult:
    admitted: bool
    reason: str = ""


class AdmissionController:
    def __init__(self, registry: PeerRegistry, gossip: GossipNode,
                 p99_latency_budget_ms: float = 500.0):
        self._reg = registry
        self._gossip = gossip
        self._budget_ms = p99_latency_budget_ms

    def check(self, node_id: str, model_id: str, shard_id: int,
              max_jobs: int = 8) -> AdmissionResult:
        entry = self._reg.get(node_id)
        if entry is None:
            return AdmissionResult(False, "unknown node")

        shard_key = f"{model_id}:{shard_id}"
        if shard_key not in entry.record.shards_hosted:
            return AdmissionResult(False, f"node does not hold shard {shard_key}")

        if entry.reputation < 200:
            return AdmissionResult(False, f"reputation too low ({entry.reputation})")

        health = self._gossip.peer_health(node_id)
        if health is None or health.is_stale():
            return AdmissionResult(False, "no recent health data")

        if health.jobs_active >= max_jobs:
            return AdmissionResult(False, f"node overloaded ({health.jobs_active} active jobs)")

        if health.avg_latency_ms > self._budget_ms:
            return AdmissionResult(
                False, f"P99 latency {health.avg_latency_ms:.0f}ms exceeds budget {self._budget_ms:.0f}ms"
            )

        return AdmissionResult(True)
