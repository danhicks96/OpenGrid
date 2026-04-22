"""
Inference engine abstraction — routes to vLLM, llama.cpp, or MLX
depending on available hardware and installed packages.

v0.0.2: Real inference implementation. Full-model-per-node for beta
(coordinator routes whole requests, not individual layer shards).
Each backend can load a model and generate text from a prompt.
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Optional

log = logging.getLogger(__name__)


class InferenceBackend(ABC):
    """Base class for all inference backends."""

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a full model from disk (GGUF, safetensors, or HF repo)."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text from a prompt. Returns the full completion."""
        ...

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Generate text token-by-token. Yields partial strings."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if a model is currently loaded and ready."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
        ...

    # Legacy interface for shard-level forward pass (post-beta: layer splitting)
    def load_shard(self, shard_path: str, shard_range: tuple[int, int]) -> None:
        self.load_model(shard_path)

    def forward(self, activations: bytes, shard_range: tuple[int, int]) -> bytes:
        """
        Shard-level forward pass. For beta (full-model-per-node), this
        interprets the input activations as a UTF-8 prompt and returns
        the generated text as bytes.
        """
        prompt = activations.decode("utf-8", errors="replace")
        output = self.generate(prompt)
        return output.encode("utf-8")


class LlamaCppBackend(InferenceBackend):
    """
    CPU/GPU inference via llama.cpp Python bindings.
    Supports GGUF models. Works on any hardware.
    Install: pip install llama-cpp-python
    """

    def __init__(self, n_threads: int = 0, n_gpu_layers: int = 0, n_ctx: int = 4096):
        self._n_threads = n_threads or (os.cpu_count() or 4)
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._model = None
        self._model_path = ""

    @staticmethod
    def _ensure_dll_paths() -> None:
        """On Windows, add nvidia/llama_cpp DLL dirs to the search path."""
        if os.name != "nt":
            return
        import site
        for sp in site.getsitepackages():
            for subdir in [
                os.path.join(sp, "nvidia", "cuda_runtime", "bin"),
                os.path.join(sp, "nvidia", "cublas", "bin"),
                os.path.join(sp, "llama_cpp", "lib"),
            ]:
                if os.path.isdir(subdir):
                    try:
                        os.add_dll_directory(subdir)
                    except OSError:
                        pass

    def load_model(self, model_path: str) -> None:
        self._ensure_dll_paths()
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            )

        log.info("Loading model: %s (threads=%d, gpu_layers=%d, ctx=%d)",
                 model_path, self._n_threads, self._n_gpu_layers, self._n_ctx)
        self._model = Llama(
            model_path=model_path,
            n_threads=self._n_threads,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            verbose=False,
        )
        self._model_path = model_path
        log.info("Model loaded: %s", model_path)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")
        result = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return result["choices"][0]["message"]["content"]

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")
        stream = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._model_path = ""


class VLLMBackend(InferenceBackend):
    """
    GPU inference via vLLM. Supports HuggingFace models + AWQ/GPTQ quantization.
    Install: pip install vllm
    """

    def __init__(self, gpu_memory_utilization: float = 0.85, max_model_len: int = 4096):
        self._gpu_util = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._engine = None
        self._tokenizer = None

    def load_model(self, model_path: str) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError:
            raise RuntimeError("vllm is not installed. Run: pip install vllm")

        log.info("Loading model via vLLM: %s (gpu_util=%.2f)", model_path, self._gpu_util)
        self._engine = LLM(
            model=model_path,
            gpu_memory_utilization=self._gpu_util,
            max_model_len=self._max_model_len,
            dtype="auto",
        )
        log.info("vLLM model loaded: %s", model_path)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        if not self._engine:
            raise RuntimeError("No model loaded. Call load_model() first.")
        from vllm import SamplingParams  # type: ignore
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        outputs = self._engine.generate([prompt], params)
        return outputs[0].outputs[0].text

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        # vLLM streaming requires the async engine; for beta, generate full then chunk
        full_output = self.generate(prompt, max_tokens, temperature, stop)
        # Yield word by word to simulate streaming
        for word in full_output.split():
            yield word + " "

    def is_loaded(self) -> bool:
        return self._engine is not None

    def unload(self) -> None:
        self._engine = None


class MLXBackend(InferenceBackend):
    """
    Apple Silicon inference via MLX. Supports HuggingFace models.
    Install: pip install mlx-lm
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def load_model(self, model_path: str) -> None:
        try:
            from mlx_lm import load  # type: ignore
        except ImportError:
            raise RuntimeError("mlx-lm not installed. Run: pip install mlx-lm")

        log.info("Loading model via MLX: %s", model_path)
        self._model, self._tokenizer = load(model_path)
        log.info("MLX model loaded: %s", model_path)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")
        from mlx_lm import generate as mlx_generate  # type: ignore
        return mlx_generate(
            self._model, self._tokenizer, prompt=prompt,
            max_tokens=max_tokens, temp=temperature,
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        full = self.generate(prompt, max_tokens, temperature, stop)
        for word in full.split():
            yield word + " "

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None


def select_backend(gpu_vram_gb: float = 0.0, platform: str = "", model_path: str = "") -> InferenceBackend:
    """Pick the best available backend for the current hardware."""
    is_gguf = model_path.lower().endswith(".gguf")

    # Apple Silicon
    if "darwin" in platform.lower() or "apple" in platform.lower():
        try:
            import mlx_lm  # type: ignore  # noqa: F401
            log.info("Selected MLX backend (Apple Silicon)")
            return MLXBackend()
        except ImportError:
            pass

    # NVIDIA GPU with enough VRAM — vLLM can't handle GGUF, force llama.cpp for those
    if gpu_vram_gb >= 6.0 and not is_gguf:
        try:
            import vllm  # type: ignore  # noqa: F401
            log.info("Selected vLLM backend (%.1f GB VRAM)", gpu_vram_gb)
            return VLLMBackend()
        except ImportError:
            log.info("GPU detected (%.1f GB) but vllm not installed; using llama.cpp", gpu_vram_gb)

    # CPU fallback (works everywhere)
    n_gpu = 0
    if gpu_vram_gb >= 4.0:
        # Offload some layers to GPU even without vLLM
        n_gpu = int(gpu_vram_gb * 3)  # rough: ~3 layers per GB
        log.info("Selected llama.cpp backend with %d GPU layers (%.1f GB VRAM)", n_gpu, gpu_vram_gb)
    else:
        log.info("Selected llama.cpp backend (CPU only)")

    return LlamaCppBackend(n_gpu_layers=n_gpu)
