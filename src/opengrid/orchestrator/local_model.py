"""
LocalModel — runs the micro orchestrator model (BitNet 2B / Phi-3-mini)
on the local CPU via llama.cpp, with tool-calling support.

This model has exactly one job: decide which tools to call to route
an inference request across the network. It is NOT the model that
generates the actual response — that's the distributed GPU nodes.

The orchestrator model is intentionally tiny (<2B params) so it runs
at 15-40 t/s on any modern CPU with no GPU required.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Optional

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are OpenGrid's local network orchestrator. Your only job is to route inference requests across a distributed network of volunteer GPU nodes.

You have access to these tools:
- find_nodes: discover which nodes can serve a model
- schedule_inference: build a pipeline DAG and assign nodes
- check_node_health: verify a node is alive and responsive
- reassign_work: recover from a node going offline mid-job
- job_status: check how a running job is progressing
- network_summary: get an overview of the current network

When a user sends a request:
1. Call network_summary to check available nodes
2. Call schedule_inference with the appropriate model
3. Return the job_id and stage assignments
4. If any node goes offline during generation, call reassign_work

You do NOT generate the actual AI response. The distributed nodes do that. Your job is coordination only. Be concise and mechanical."""


class LocalOrchestratorModel:
    """
    Wraps llama.cpp with tool-calling to drive network orchestration.
    Falls back to a rule-based orchestrator if llama.cpp is not installed
    or no model file is available.
    """

    def __init__(self, model_path: Optional[Path] = None, n_threads: int = 4):
        self._model_path = model_path
        self._n_threads = n_threads
        self._llm = None
        self._load()

    def _load(self) -> None:
        if self._model_path is None or not self._model_path.exists():
            log.warning(
                "No local orchestrator model found at %s — using rule-based fallback.",
                self._model_path,
            )
            return
        try:
            from llama_cpp import Llama  # type: ignore
            self._llm = Llama(
                model_path=str(self._model_path),
                n_threads=self._n_threads,
                n_ctx=2048,
                verbose=False,
            )
            log.info("Local orchestrator model loaded: %s", self._model_path.name)
        except ImportError:
            log.warning("llama-cpp-python not installed — using rule-based fallback.")
        except Exception as e:
            log.warning("Failed to load orchestrator model: %s", e)

    def is_loaded(self) -> bool:
        return self._llm is not None

    def decide(self, user_message: str, tools: "OrchestratorTools") -> list[dict]:  # noqa: F821
        """
        Given a user message, return the sequence of tool calls needed to
        route the request. Returns list of {name, arguments, result} dicts.
        """
        if self._llm is not None:
            return self._llm_decide(user_message, tools)
        return self._rule_based_decide(user_message, tools)

    def _llm_decide(self, user_message: str, tools) -> list[dict]:
        """Use the local model to decide tool calls via llama.cpp tool_choice."""
        from llama_cpp import Llama  # type: ignore
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        calls = []
        # Agentic loop: keep calling until no more tool calls
        for _ in range(6):  # max 6 tool call rounds
            response = self._llm.create_chat_completion(
                messages=messages,
                tools=[{"type": "function", "function": t} for t in tools.SCHEMA],
                tool_choice="auto",
                max_tokens=512,
                temperature=0.0,
            )
            choice = response["choices"][0]
            msg = choice["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break

            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    args = {}
                result = tools.call(name, args)
                calls.append({"name": name, "arguments": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })

        return calls

    def _rule_based_decide(self, user_message: str, tools) -> list[dict]:
        """
        Deterministic fallback: always schedule on the default model.
        No LLM required — just calls the tools directly in order.
        """
        calls = []

        # Step 1: network summary
        summary = tools.call("network_summary", {})
        calls.append({"name": "network_summary", "arguments": {}, "result": summary})

        if summary.get("active_peers", 0) == 0:
            return calls  # no nodes available — caller handles this

        # Step 2: pick model (prefer 70B if heavy/power nodes exist, else 8B, else bitnet)
        tiers = summary.get("tiers", {})
        if tiers.get("heavy", 0) + tiers.get("power", 0) >= 2:
            model_id = "llama3-70b-int4"
        elif tiers.get("mid", 0) >= 1:
            model_id = "llama3-8b-int4"
        else:
            model_id = "bitnet-b158-2b"

        # Step 3: schedule
        sched = tools.call("schedule_inference", {"model_id": model_id})
        calls.append({"name": "schedule_inference", "arguments": {"model_id": model_id}, "result": sched})

        return calls
