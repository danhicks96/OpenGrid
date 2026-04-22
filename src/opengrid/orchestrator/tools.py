"""
Orchestrator tool definitions — the micro model calls these to manage
distributed inference across the network.

Each tool is a plain Python function. The orchestrator model invokes them
via JSON tool-call format (llama.cpp tool_choice support).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Any, Optional

log = logging.getLogger(__name__)


class OrchestratorTools:
    """
    Stateful tool container bound to the live network components.
    Injected into the orchestrator at startup.
    """

    def __init__(self, scheduler, peer_registry, gossip, work_monitor, kv_router):
        self._scheduler = scheduler
        self._peers = peer_registry
        self._gossip = gossip
        self._monitor = work_monitor
        self._kv_router = kv_router

    # ── Tool: find_nodes ────────────────────────────────────────────────────

    def find_nodes(self, model_id: str, shard_id: Optional[int] = None) -> dict:
        """
        Find live nodes that can serve a given model (and optionally a specific shard).
        Returns list of node summaries.
        """
        if shard_id is not None:
            entries = self._peers.with_shard(model_id, shard_id)
        else:
            entries = self._peers.all_active()

        return {
            "nodes": [
                {
                    "node_id": e.record.node_id,
                    "tier": e.record.tier,
                    "shards": e.record.shards_hosted,
                    "reputation": e.reputation,
                }
                for e in entries[:20]  # cap result size
            ],
            "count": len(entries),
        }

    # ── Tool: schedule_inference ────────────────────────────────────────────

    def schedule_inference(self, model_id: str, session_id: Optional[str] = None) -> dict:
        """
        Build a full pipeline DAG for the given model.
        Returns job_id and per-stage node assignments.
        """
        sid = session_id or str(uuid.uuid4())
        result = self._scheduler.schedule(model_id, session_id=sid)
        if result.dag is None:
            return {"error": result.error}

        self._monitor.register_job(result.dag)
        return {
            "job_id": result.dag.job_id,
            "session_id": sid,
            "stages": [
                {
                    "task_id": t.task_id,
                    "node_id": t.node_id,
                    "shard_range": list(t.shard_range),
                }
                for t in result.dag.tasks
            ],
        }

    # ── Tool: check_node_health ─────────────────────────────────────────────

    def check_node_health(self, node_id: str) -> dict:
        """
        Check the latest health report for a specific node.
        """
        health = self._gossip.peer_health(node_id)
        if health is None:
            return {"status": "unknown", "node_id": node_id}
        return {
            "node_id": node_id,
            "status": health.status,
            "vram_free_gb": health.vram_free_gb,
            "jobs_active": health.jobs_active,
            "avg_latency_ms": health.avg_latency_ms,
            "stale": health.is_stale(),
        }

    # ── Tool: reassign_work ─────────────────────────────────────────────────

    def reassign_work(self, job_id: str, failed_node_id: str) -> dict:
        """
        Reassign all tasks owned by failed_node_id for job_id to backup nodes.
        Resumes from the last saved activation checkpoint.
        """
        result = self._monitor.reassign(job_id, failed_node_id)
        return result

    # ── Tool: job_status ────────────────────────────────────────────────────

    def job_status(self, job_id: str) -> dict:
        """
        Return current status of all tasks in a job DAG.
        """
        return self._monitor.status(job_id)

    # ── Tool: network_summary ───────────────────────────────────────────────

    def network_summary(self) -> dict:
        """
        High-level view of the network: peer count, tier breakdown, active jobs.
        """
        peers = self._gossip.active_peers()
        tier_counts: dict[str, int] = {}
        for p in peers:
            tier_counts[p.tier] = tier_counts.get(p.tier, 0) + 1
        return {
            "active_peers": len(peers),
            "tiers": tier_counts,
            "active_jobs": self._monitor.active_job_count(),
        }

    # ── Dispatch table ──────────────────────────────────────────────────────

    SCHEMA = [
        {
            "name": "find_nodes",
            "description": "Find live nodes that hold shards for a model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "e.g. llama3-70b-int4"},
                    "shard_id": {"type": "integer", "description": "Optional specific shard index"},
                },
                "required": ["model_id"],
            },
        },
        {
            "name": "schedule_inference",
            "description": "Build a full pipeline DAG and assign nodes for distributed inference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "session_id": {"type": "string", "description": "Optional: reuse existing KV cache session"},
                },
                "required": ["model_id"],
            },
        },
        {
            "name": "check_node_health",
            "description": "Get latest health data for a specific node.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
        {
            "name": "reassign_work",
            "description": "Reassign failed node's work units to a backup node, resuming from last checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "failed_node_id": {"type": "string"},
                },
                "required": ["job_id", "failed_node_id"],
            },
        },
        {
            "name": "job_status",
            "description": "Get the current status of all tasks in a running job.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
        {
            "name": "network_summary",
            "description": "Get a high-level summary of the current network state.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    def call(self, name: str, arguments: dict) -> Any:
        fn = getattr(self, name, None)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(**arguments)
        except Exception as e:
            log.exception("Tool %s failed", name)
            return {"error": str(e)}
