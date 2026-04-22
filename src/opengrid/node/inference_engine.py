"""
Inference engine abstraction — routes to vLLM, llama.cpp, or MLX
depending on available hardware and installed packages.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

log = logging.getLogger(__name__)


class InferenceBackend(ABC):
    @abstractmethod
    def load_shard(self, shard_path: str, shard_range: tuple[int, int]) -> None: ...

    @abstractmethod
    def forward(self, activations: bytes, shard_range: tuple[int, int]) -> bytes: ...

    @abstractmethod
    def unload(self) -> None: ...


class VLLMBackend(InferenceBackend):
    """GPU INT4/INT8 via vLLM.  Requires: pip install vllm"""

    def __init__(self, model_id: str, gpu_memory_utilization: float = 0.90):
        self.model_id = model_id
        self._gpu_util = gpu_memory_utilization
        self._engine = None

    def load_shard(self, shard_path: str, shard_range: tuple[int, int]) -> None:
        try:
            from vllm import LLM  # type: ignore
            self._engine = LLM(
                model=shard_path,
                gpu_memory_utilization=self._gpu_util,
                dtype="auto",
            )
            log.info("vLLM loaded shard %s layers %s", shard_path, shard_range)
        except ImportError:
            raise RuntimeError("vllm is not installed. Run: pip install vllm")

    def forward(self, activations: bytes, shard_range: tuple[int, int]) -> bytes:
        # In a real impl this calls the vLLM custom forward pass for the layer range
        raise NotImplementedError("vLLM shard-level forward not yet wired")

    def unload(self) -> None:
        self._engine = None


class LlamaCppBackend(InferenceBackend):
    """CPU INT4 via llama.cpp Python bindings.  Requires: pip install llama-cpp-python"""

    def __init__(self, n_threads: int = 4):
        self._n_threads = n_threads
        self._model = None

    def load_shard(self, shard_path: str, shard_range: tuple[int, int]) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
            self._model = Llama(
                model_path=shard_path,
                n_threads=self._n_threads,
                verbose=False,
            )
            log.info("llama.cpp loaded %s", shard_path)
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

    def forward(self, activations: bytes, shard_range: tuple[int, int]) -> bytes:
        raise NotImplementedError("llama.cpp shard-level forward not yet wired")

    def unload(self) -> None:
        self._model = None


class MLXBackend(InferenceBackend):
    """Apple Silicon via MLX.  Requires: pip install mlx-lm"""

    def load_shard(self, shard_path: str, shard_range: tuple[int, int]) -> None:
        try:
            import mlx_lm  # type: ignore  # noqa: F401
            log.info("MLX backend loaded %s", shard_path)
        except ImportError:
            raise RuntimeError("mlx-lm not installed. Run: pip install mlx-lm")

    def forward(self, activations: bytes, shard_range: tuple[int, int]) -> bytes:
        raise NotImplementedError("MLX shard-level forward not yet wired")

    def unload(self) -> None:
        pass


def select_backend(gpu_vram_gb: float = 0.0, platform: str = "") -> InferenceBackend:
    """Pick the best available backend for the current hardware."""
    if "darwin" in platform.lower() or "apple" in platform.lower():
        try:
            import mlx_lm  # type: ignore  # noqa: F401
            return MLXBackend()
        except ImportError:
            pass

    if gpu_vram_gb >= 6.0:
        try:
            import vllm  # type: ignore  # noqa: F401
            return VLLMBackend(model_id="")
        except ImportError:
            log.warning("GPU detected but vllm not installed; falling back to llama.cpp")

    return LlamaCppBackend()
