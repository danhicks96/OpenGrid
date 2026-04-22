"""
Reputation scoring system for worker nodes.
Scores run 0–1000; nodes start at 500.
"""
from __future__ import annotations

from opengrid.mesh.peer_registry import PeerRegistry

SCORE_PROOF_PASS = +1
SCORE_PROOF_FAIL_SOFT = -5
SCORE_PROOF_FAIL_HARD = -200
SCORE_TIMEOUT_NOT_FAULT = -2
SCORE_TIMEOUT_FAULT = -20
SCORE_UPTIME_BONUS = +10      # awarded per 24 h of active uptime

DEMOTION_THRESHOLD = 200      # below this → validation-only tasks
BAN_THRESHOLD = 100           # below this → 24 h temp ban
PROBATION_TASK_COUNT = 1000   # new nodes need this many validated tasks


class ReputationManager:
    def __init__(self, registry: PeerRegistry):
        self._reg = registry

    def proof_passed(self, node_id: str) -> None:
        self._reg.adjust_reputation(node_id, SCORE_PROOF_PASS)

    def proof_failed_soft(self, node_id: str) -> None:
        self._reg.adjust_reputation(node_id, SCORE_PROOF_FAIL_SOFT)

    def proof_failed_hard(self, node_id: str) -> None:
        self._reg.adjust_reputation(node_id, SCORE_PROOF_FAIL_HARD)

    def timeout_not_fault(self, node_id: str) -> None:
        self._reg.adjust_reputation(node_id, SCORE_TIMEOUT_NOT_FAULT)

    def timeout_fault(self, node_id: str) -> None:
        self._reg.adjust_reputation(node_id, SCORE_TIMEOUT_FAULT)

    def uptime_bonus(self, node_id: str) -> None:
        self._reg.adjust_reputation(node_id, SCORE_UPTIME_BONUS)

    def is_eligible(self, node_id: str, high_value: bool = False) -> bool:
        entry = self._reg.get(node_id)
        if entry is None:
            return False
        score = entry.reputation
        if score < BAN_THRESHOLD:
            return False
        if score < DEMOTION_THRESHOLD:
            return False  # demoted to validation-only — handled by caller
        if high_value and entry.jobs_completed < PROBATION_TASK_COUNT:
            return False
        return True

    def score(self, node_id: str) -> int:
        entry = self._reg.get(node_id)
        return entry.reputation if entry else 0
