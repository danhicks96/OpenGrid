"""
Pipeline executor — chains sharded inference across multiple nodes.

Takes a prompt, breaks it into a pipeline of shard stages, sends
activations between nodes, and assembles the final token output.

This is the 1:N distributed inference that makes OpenGrid work.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """One stage of the inference pipeline."""
    node_id: str
    host: str
    port: int
    layer_start: int
    layer_end: int
    is_first: bool = False
    is_last: bool = False


@dataclass
class PipelineResult:
    output_text: str = ""
    tokens_generated: int = 0
    total_latency_ms: float = 0.0
    stages_completed: int = 0
    error: str = ""


class ShardedPipeline:
    """
    Orchestrates multi-node sharded inference.

    For local testing (single machine, simulated multi-shard):
        pipeline = ShardedPipeline()
        pipeline.add_local_stage(shard_backend_0_5)
        pipeline.add_local_stage(shard_backend_6_10)
        pipeline.add_local_stage(shard_backend_11_16)
        pipeline.add_local_stage(shard_backend_17_21)
        result = pipeline.run("What is 2+2?")

    For distributed (multi-machine):
        pipeline = ShardedPipeline()
        pipeline.add_remote_stage("node-abc", "192.168.1.10", 7610, 0, 5)
        pipeline.add_remote_stage("node-def", "192.168.1.11", 7610, 6, 10)
        ...
        result = await pipeline.run_distributed("What is 2+2?")
    """

    def __init__(self):
        self._local_stages: list = []   # list of ShardedBackend instances
        self._remote_stages: list[PipelineStage] = []

    def add_local_stage(self, backend) -> None:
        """Add a local ShardedBackend as a pipeline stage."""
        self._local_stages.append(backend)

    def add_remote_stage(
        self, node_id: str, host: str, port: int,
        layer_start: int, layer_end: int,
        is_first: bool = False, is_last: bool = False,
    ) -> None:
        self._remote_stages.append(PipelineStage(
            node_id=node_id, host=host, port=port,
            layer_start=layer_start, layer_end=layer_end,
            is_first=is_first, is_last=is_last,
        ))

    def run_local(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
    ) -> PipelineResult:
        """
        Run the full pipeline locally using multiple ShardedBackend instances.
        Each backend holds a different layer range.
        Proves the shard splitting works before going distributed.
        """
        if not self._local_stages:
            return PipelineResult(error="No local stages configured")

        t0 = time.perf_counter()
        first = self._local_stages[0]
        last = self._local_stages[-1]

        try:
            # Tokenize and embed (first shard)
            input_ids = first.tokenize(prompt)
            hidden = first.embed(input_ids)

            # Forward through all stages
            for i, stage in enumerate(self._local_stages):
                hidden = stage.forward_layers(hidden)
                log.debug("Stage %d (layers %d-%d) complete, activation shape: %s",
                         i, stage.layer_start, stage.layer_end, hidden.shape)

            # Decode tokens autoregressively (last shard)
            generated = []
            for step in range(max_tokens):
                token_id, _ = last.decode_next_token(hidden, temperature)
                if token_id == first._tokenizer.eos_token_id:
                    break
                generated.append(token_id)

                # Next step: embed new token → run through ALL stages again
                new_embed = first.embed(
                    first._tokenizer.encode(
                        first._tokenizer.decode([token_id]),
                        return_tensors="pt"
                    ).to(first._device)
                )
                for stage in self._local_stages:
                    new_embed = stage.forward_layers(new_embed)
                hidden = new_embed

            output_text = first._tokenizer.decode(generated, skip_special_tokens=True)
            latency = (time.perf_counter() - t0) * 1000

            return PipelineResult(
                output_text=output_text,
                tokens_generated=len(generated),
                total_latency_ms=round(latency, 2),
                stages_completed=len(self._local_stages),
            )

        except Exception as e:
            log.exception("Pipeline execution failed")
            return PipelineResult(
                error=str(e),
                total_latency_ms=(time.perf_counter() - t0) * 1000,
            )

    @property
    def stage_count(self) -> int:
        return len(self._local_stages) + len(self._remote_stages)
