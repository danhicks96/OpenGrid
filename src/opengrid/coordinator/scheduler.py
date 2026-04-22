"""
Request scheduler — translates an inference request into a pipeline DAG
and assigns each stage to an eligible worker node.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from opengrid.coordinator.admission import AdmissionController
from opengrid.coordinator.dag_executor import InferenceDAG, build_pipeline_dag
from opengrid.coordinator.kv_router import KVRouter
from opengrid.mesh.peer_registry import PeerRegistry
from opengrid.registry.model_registry import ModelRegistry

log = logging.getLogger(__name__)

MODEL_COST_FACTORS: dict[str, float] = {
    "bitnet-b158-2b": 0.25,
    "llama3-8b-int4": 1.0,
    "llama3-70b-int4": 2.5,
    "mixtral-8x7b-int4": 2.0,
}


@dataclass
class ScheduleResult:
    dag: Optional[InferenceDAG]
    error: str = ""


class Scheduler:
    def __init__(
        self,
        registry: PeerRegistry,
        model_registry: ModelRegistry,
        admission: AdmissionController,
        kv_router: KVRouter,
        deadline_ms: int = 2000,
    ):
        self._peers = registry
        self._models = model_registry
        self._admission = admission
        self._kv_router = kv_router
        self._deadline_ms = deadline_ms

    def schedule(self, model_id: str, session_id: Optional[str] = None) -> ScheduleResult:
        manifest = self._models.get(model_id)
        if manifest is None:
            return ScheduleResult(None, f"Unknown model: {model_id}")

        assignments: list[tuple[tuple[int, int], str]] = []

        for shard in manifest.shards:
            shard_range = tuple(shard.layers)  # [first, last]
            # Check KV-cache-warm node first
            preferred = None
            if session_id:
                preferred = self._kv_router.preferred_node(session_id)
                if preferred:
                    result = self._admission.check(preferred, model_id, shard.shard_id)
                    if not result.admitted:
                        preferred = None

            node_id = preferred or self._find_node(model_id, shard.shard_id)
            if node_id is None:
                return ScheduleResult(
                    None, f"No eligible node for shard {shard.shard_id} of {model_id}"
                )
            assignments.append((shard_range, node_id))

        dag = build_pipeline_dag(
            job_id=str(uuid.uuid4()),
            model_id=model_id,
            shard_assignments=assignments,
            deadline_ms=self._deadline_ms,
        )
        return ScheduleResult(dag=dag)

    def _find_node(self, model_id: str, shard_id: int) -> Optional[str]:
        candidates = self._peers.with_shard(model_id, shard_id)
        # Sort by reputation desc, then by jobs_active asc
        candidates.sort(key=lambda e: (-e.reputation, e.record.node_id))
        for entry in candidates:
            result = self._admission.check(entry.record.node_id, model_id, shard_id)
            if result.admitted:
                return entry.record.node_id
        return None
