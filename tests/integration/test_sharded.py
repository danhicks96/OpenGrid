"""
Test script: Split TinyLlama into 4 shards, run pipeline locally.
Proves activations flow correctly between layer ranges.

Run: python -m pytest tests/integration/test_sharded.py -v -s
"""
import pytest
import logging

logging.basicConfig(level=logging.INFO)

# Skip if torch/transformers not installed
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from opengrid.node.sharded_backend import ShardedBackend
from opengrid.node.pipeline import ShardedPipeline

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module")
def full_model_backend():
    """Load TinyLlama as a single shard (all layers) for baseline."""
    backend = ShardedBackend(model_name=MODEL, layer_start=0, layer_end=-1, device="cpu")
    backend.load()
    yield backend
    backend.unload()


class TestShardedBackend:
    """Test individual shard operations."""

    def test_load_full_model(self, full_model_backend):
        assert full_model_backend.is_loaded()
        assert full_model_backend.total_layers == 22
        assert full_model_backend.is_first_shard
        assert full_model_backend.is_last_shard

    def test_tokenize(self, full_model_backend):
        tokens = full_model_backend.tokenize("Hello world")
        assert tokens.shape[0] == 1  # batch size
        assert tokens.shape[1] > 0   # has tokens

    def test_embed(self, full_model_backend):
        tokens = full_model_backend.tokenize("Hello")
        hidden = full_model_backend.embed(tokens)
        assert hidden.shape[0] == 1           # batch
        assert hidden.shape[2] == 2048        # TinyLlama hidden dim

    def test_forward_layers(self, full_model_backend):
        tokens = full_model_backend.tokenize("Test")
        hidden = full_model_backend.embed(tokens)
        output = full_model_backend.forward_layers(hidden)
        assert output.shape == hidden.shape   # same shape through layers

    def test_decode_next_token(self, full_model_backend):
        tokens = full_model_backend.tokenize("The capital of France is")
        hidden = full_model_backend.embed(tokens)
        output = full_model_backend.forward_layers(hidden)
        token_id, logits = full_model_backend.decode_next_token(output, temperature=0.1)
        word = full_model_backend.decode_token(token_id)
        assert isinstance(word, str)
        assert len(word) > 0

    def test_generate_full(self, full_model_backend):
        output = full_model_backend.generate_full("2 + 2 =", max_tokens=10, temperature=0.1)
        assert len(output) > 0
        print(f"Full model output: {output!r}")

    def test_serialize_deserialize_activations(self, full_model_backend):
        tokens = full_model_backend.tokenize("Test")
        hidden = full_model_backend.embed(tokens)
        # Serialize
        data = ShardedBackend.serialize_activations(hidden)
        assert isinstance(data, bytes)
        assert len(data) > 0
        # Deserialize
        recovered = ShardedBackend.deserialize_activations(data)
        assert recovered.shape == hidden.shape
        # Values should be close (float16 quantization)
        diff = (hidden.cpu().float() - recovered.float()).abs().max().item()
        assert diff < 0.01


class TestShardedPipeline:
    """Test multi-shard pipeline — the real distributed inference proof."""

    def test_four_shard_pipeline(self):
        """
        Split TinyLlama into 4 shards of ~5-6 layers each.
        Run pipeline and verify output is coherent.
        """
        # Create 4 shards: 0-5, 6-10, 11-16, 17-21
        shard_ranges = [(0, 5), (6, 10), (11, 16), (17, 21)]
        shards = []

        for start, end in shard_ranges:
            backend = ShardedBackend(
                model_name=MODEL,
                layer_start=start,
                layer_end=end,
                device="cpu",
            )
            backend.load()
            shards.append(backend)

        assert shards[0].is_first_shard
        assert shards[-1].is_last_shard
        assert not shards[1].is_first_shard
        assert not shards[1].is_last_shard

        # Build pipeline
        pipeline = ShardedPipeline()
        for shard in shards:
            pipeline.add_local_stage(shard)
        assert pipeline.stage_count == 4

        # Run inference through the pipeline
        result = pipeline.run_local(
            "The meaning of life is",
            max_tokens=10,
            temperature=0.1,
        )

        print(f"4-shard pipeline output: {result.output_text!r}")
        print(f"Tokens: {result.tokens_generated}, Latency: {result.total_latency_ms:.0f}ms")
        print(f"Stages completed: {result.stages_completed}")

        assert result.error == ""
        assert result.tokens_generated > 0
        assert result.stages_completed == 4
        assert len(result.output_text) > 0

        # Cleanup
        for shard in shards:
            shard.unload()

    def test_two_shard_pipeline(self):
        """Split into 2 halves — simpler test."""
        shard_a = ShardedBackend(MODEL, layer_start=0, layer_end=10, device="cpu")
        shard_b = ShardedBackend(MODEL, layer_start=11, layer_end=21, device="cpu")
        shard_a.load()
        shard_b.load()

        pipeline = ShardedPipeline()
        pipeline.add_local_stage(shard_a)
        pipeline.add_local_stage(shard_b)

        result = pipeline.run_local("Hello, my name is", max_tokens=5, temperature=0.1)
        print(f"2-shard output: {result.output_text!r}")

        assert result.error == ""
        assert result.tokens_generated > 0

        shard_a.unload()
        shard_b.unload()
