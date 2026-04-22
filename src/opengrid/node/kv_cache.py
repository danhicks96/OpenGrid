"""
KV cache store — per-session key-value attention cache with LRU eviction.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict
from threading import RLock
from typing import Optional

log = logging.getLogger(__name__)


class KVCacheStore:
    def __init__(self, max_ram_gb: float = 4.0):
        self._max_bytes = int(max_ram_gb * 1e9)
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._used = 0
        self._lock = RLock()

    def _key(self, job_id: str, shard_range: tuple[int, int]) -> str:
        return f"{job_id}:{shard_range[0]}-{shard_range[1]}"

    def put(self, job_id: str, shard_range: tuple[int, int], data: bytes) -> None:
        k = self._key(job_id, shard_range)
        size = len(data)
        with self._lock:
            # Evict LRU entries until we have room
            while self._used + size > self._max_bytes and self._cache:
                evict_key, _ = self._cache.popitem(last=False)
                freed = self._sizes.pop(evict_key, 0)
                self._used -= freed
                log.debug("KV evict %s (freed %d bytes)", evict_key, freed)

            if size > self._max_bytes:
                log.warning("KV entry %s (%d bytes) exceeds total budget; skipping", k, size)
                return

            if k in self._cache:
                self._used -= self._sizes[k]
            self._cache[k] = data
            self._cache.move_to_end(k)
            self._sizes[k] = size
            self._used += size

    def get(self, job_id: str, shard_range: tuple[int, int]) -> Optional[bytes]:
        k = self._key(job_id, shard_range)
        with self._lock:
            if k not in self._cache:
                return None
            self._cache.move_to_end(k)
            return self._cache[k]

    def evict_job(self, job_id: str) -> None:
        with self._lock:
            keys = [k for k in self._cache if k.startswith(f"{job_id}:")]
            for k in keys:
                del self._cache[k]
                self._used -= self._sizes.pop(k, 0)

    def used_bytes(self) -> int:
        return self._used

    def used_gb(self) -> float:
        return round(self._used / 1e9, 3)
