"""
KV-cache-aware router — routes decode steps to nodes holding the session's KV cache.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

from opengrid.mesh.peer_registry import PeerRegistry

log = logging.getLogger(__name__)


class KVRouter:
    def __init__(self, registry: PeerRegistry, ttl_seconds: float = 3600.0):
        # session_id → (node_id, expires_at)
        self._session_map: dict[str, tuple[str, float]] = {}
        self._registry = registry
        self._ttl = ttl_seconds

    def record_session(self, session_id: str, node_id: str) -> None:
        self._session_map[session_id] = (node_id, time.time() + self._ttl)

    def preferred_node(self, session_id: str) -> Optional[str]:
        """Return the node holding this session's KV cache, if still live."""
        mapping = self._session_map.get(session_id)
        if mapping is None:
            return None
        node_id, expires_at = mapping
        if time.time() > expires_at:
            del self._session_map[session_id]
            return None
        entry = self._registry.get(node_id)
        if entry is None or not entry.is_alive():
            del self._session_map[session_id]
            return None
        return node_id

    def evict_session(self, session_id: str) -> None:
        self._session_map.pop(session_id, None)

    def prefix_hash(self, messages: list[dict]) -> str:
        """Hash a message list to use as a cache key prefix."""
        blob = str(messages).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
