"""
ShardedBackend — PyTorch-based layer-range inference for distributed
pipeline parallelism.

Each node loads only its assigned transformer layers into VRAM/RAM.
Accepts intermediate activations as input, runs its layer range,
outputs activations for the next node in the pipeline.

This is the backend that makes "split a 70B model across 4 gaming PCs"
actually work. llama.cpp can't do this — it runs full models only.

Usage:
    backend = ShardedBackend(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        layer_start=6,
        layer_end=10,    # exclusive — runs layers 6,7,8,9,10
        device="cuda",   # or "cpu"
    )
    backend.load()

    # First shard (layer_start=0) — tokenize input
    input_ids = backend.tokenize("What is 2+2?")
    activations = backend.embed(input_ids)

    # Middle shards — run layer range on activations
    activations = backend.forward_layers(activations)

    # Last shard — run layers + decode to text
    tokens = backend.decode(activations)
"""
from __future__ import annotations

import gc
import io
import logging
import struct
import time
from typing import Iterator, Optional

import torch

log = logging.getLogger(__name__)


class ShardedBackend:
    """
    Loads a specific range of transformer layers from a HuggingFace model
    and runs partial forward passes on intermediate activations.
    """

    def __init__(
        self,
        model_name: str = "",
        layer_start: int = 0,
        layer_end: int = -1,        # -1 = last layer
        device: str = "auto",
        dtype: str = "auto",        # "float16", "bfloat16", "float32", "auto"
        max_memory_gb: float = 0.0, # 0 = no limit
    ):
        self.model_name = model_name
        self.layer_start = layer_start
        self.layer_end = layer_end
        self._device_str = device
        self._dtype_str = dtype
        self._max_memory_gb = max_memory_gb

        self._model = None
        self._tokenizer = None
        self._layers = None
        self._embed_tokens = None
        self._norm = None
        self._lm_head = None
        self._rotary_emb = None
        self._total_layers = 0
        self._is_first_shard = False
        self._is_last_shard = False
        self._device = None
        self._dtype = None

    def load(self) -> None:
        """Load model and extract only the assigned layer range."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers not installed. Run: pip install transformers torch"
            )

        log.info("ShardedBackend loading %s (layers %d-%d)...",
                 self.model_name, self.layer_start, self.layer_end)

        # Determine device
        if self._device_str == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._device_str)

        # Determine dtype
        if self._dtype_str == "auto":
            if self._device.type == "cuda":
                self._dtype = torch.float16
            else:
                self._dtype = torch.float32
        else:
            self._dtype = getattr(torch, self._dtype_str)

        # Load tokenizer (lightweight, every shard needs it for the first/last shard)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load full model to CPU first, then extract layers
        # For large models, use device_map="cpu" to avoid OOM during load
        log.info("Loading full model to CPU for layer extraction...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        # Find the transformer layers (handles different model architectures)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA, Mistral, TinyLlama, Qwen, etc.
            all_layers = model.model.layers
            self._embed_tokens = model.model.embed_tokens
            self._norm = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2, GPT-Neo, etc.
            all_layers = model.transformer.h
            self._embed_tokens = model.transformer.wte
            self._norm = model.transformer.ln_f
        else:
            raise RuntimeError(
                f"Unknown model architecture: {type(model).__name__}. "
                "ShardedBackend supports LLaMA-family and GPT-family models."
            )

        self._total_layers = len(all_layers)
        if self.layer_end == -1:
            self.layer_end = self._total_layers - 1

        self._is_first_shard = (self.layer_start == 0)
        self._is_last_shard = (self.layer_end >= self._total_layers - 1)

        log.info("Model has %d layers. This shard: layers %d-%d (%s%s)",
                 self._total_layers, self.layer_start, self.layer_end,
                 "FIRST " if self._is_first_shard else "",
                 "LAST" if self._is_last_shard else "")

        # Extract only our layers and move to device
        self._layers = torch.nn.ModuleList([
            all_layers[i] for i in range(self.layer_start, self.layer_end + 1)
        ]).to(self._device)

        # Extract rotary embedding (needed by ALL shards for position encoding)
        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            self._rotary_emb = model.model.rotary_emb.to(self._device)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "rotary_emb"):
            self._rotary_emb = model.transformer.rotary_emb.to(self._device)

        # First shard needs embedding layer
        if self._is_first_shard:
            self._embed_tokens = self._embed_tokens.to(self._device)
        else:
            self._embed_tokens = None

        # Last shard needs norm + lm_head
        if self._is_last_shard:
            self._norm = self._norm.to(self._device)
            if hasattr(model, "lm_head"):
                self._lm_head = model.lm_head.to(self._device)
            elif hasattr(model, "transformer") and hasattr(model.transformer, "lm_head"):
                self._lm_head = model.transformer.lm_head.to(self._device)
        else:
            self._norm = None
            self._lm_head = None

        # Free the full model from memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log.info("Shard loaded. Layers: %d, Device: %s, VRAM used: %.2f GB",
                 len(self._layers), self._device, vram_used)

    def is_loaded(self) -> bool:
        return self._layers is not None

    def unload(self) -> None:
        self._layers = None
        self._embed_tokens = None
        self._norm = None
        self._lm_head = None
        self._rotary_emb = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Tokenization (first shard only) ───────────────────────────────────

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text. Only needed by the first shard."""
        tokens = self._tokenizer.encode(text, return_tensors="pt")
        return tokens.to(self._device)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings. Only the first shard does this."""
        if self._embed_tokens is None:
            raise RuntimeError("embed() called on non-first shard")
        with torch.no_grad():
            return self._embed_tokens(input_ids)

    # ── Layer-range forward pass ──────────────────────────────────────────

    @torch.no_grad()
    def forward_layers(self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Run this shard's layer range on intermediate activations.
        Input: hidden_states from previous shard (or from embed() if first)
        Output: hidden_states for next shard (or for decode() if last)
        """
        if self._layers is None:
            raise RuntimeError("Shard not loaded. Call load() first.")

        hidden_states = hidden_states.to(self._device)

        # Generate position IDs if not provided
        seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=self._device).unsqueeze(0)

        # Compute rotary position embeddings (RoPE)
        position_embeddings = None
        if self._rotary_emb is not None:
            position_embeddings = self._rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(self._layers):
            try:
                layer_output = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                )
            except TypeError:
                # Fallback: try without position_embeddings
                try:
                    layer_output = layer(hidden_states, position_ids=position_ids)
                except TypeError:
                    layer_output = layer(hidden_states)
            # Handle different return types
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        return hidden_states

    # ── Token decoding (last shard only) ──────────────────────────────────

    @torch.no_grad()
    def decode_next_token(
        self, hidden_states: torch.Tensor, temperature: float = 0.7
    ) -> tuple[int, torch.Tensor]:
        """
        Convert final hidden states to a token ID. Only the last shard does this.
        Returns (token_id, logits).
        """
        if self._norm is None or self._lm_head is None:
            raise RuntimeError("decode_next_token() called on non-last shard")

        normed = self._norm(hidden_states)
        logits = self._lm_head(normed)

        # Sample from last position
        next_logits = logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
        else:
            token_id = next_logits.argmax(dim=-1).item()

        return token_id, logits

    def decode_token(self, token_id: int) -> str:
        """Convert a token ID back to text."""
        return self._tokenizer.decode([token_id])

    # ── Full generation (for single-node testing) ─────────────────────────

    @torch.no_grad()
    def generate_full(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """
        Run the full pipeline on a single node (for testing).
        Only works if this shard holds ALL layers (start=0, end=-1).
        """
        if not self._is_first_shard or not self._is_last_shard:
            raise RuntimeError(
                "generate_full() requires all layers. "
                f"This shard has layers {self.layer_start}-{self.layer_end} "
                f"of {self._total_layers}."
            )

        input_ids = self.tokenize(prompt)
        hidden = self.embed(input_ids)
        hidden = self.forward_layers(hidden)

        generated = []
        for _ in range(max_tokens):
            token_id, _ = self.decode_next_token(hidden, temperature)
            if token_id == self._tokenizer.eos_token_id:
                break
            generated.append(token_id)
            # For autoregressive: re-embed the new token and run through all layers
            new_embed = self._embed_tokens(
                torch.tensor([[token_id]], device=self._device)
            )
            hidden = self.forward_layers(new_embed)

        return self._tokenizer.decode(generated)

    # ── Activation serialization ──────────────────────────────────────────

    @staticmethod
    def serialize_activations(tensor: torch.Tensor) -> bytes:
        """Serialize a tensor to bytes for network transfer."""
        buf = io.BytesIO()
        # Header: shape dims + dtype
        shape = tensor.shape
        buf.write(struct.pack("I", len(shape)))
        for dim in shape:
            buf.write(struct.pack("I", dim))
        # Convert to CPU float16 for transfer (saves bandwidth)
        cpu_tensor = tensor.cpu().half()
        buf.write(cpu_tensor.numpy().tobytes())
        return buf.getvalue()

    @staticmethod
    def deserialize_activations(data: bytes, device: str = "cpu") -> torch.Tensor:
        """Deserialize bytes back to a tensor."""
        buf = io.BytesIO(data)
        n_dims = struct.unpack("I", buf.read(4))[0]
        shape = tuple(struct.unpack("I", buf.read(4))[0] for _ in range(n_dims))
        import numpy as np
        arr = np.frombuffer(buf.read(), dtype=np.float16).reshape(shape)
        return torch.from_numpy(arr.copy()).to(device)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_first_shard(self) -> bool:
        return self._is_first_shard

    @property
    def is_last_shard(self) -> bool:
        return self._is_last_shard

    @property
    def total_layers(self) -> int:
        return self._total_layers

    @property
    def shard_layer_count(self) -> int:
        return len(self._layers) if self._layers else 0
