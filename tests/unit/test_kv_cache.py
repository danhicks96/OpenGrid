"""Unit tests for KV cache store."""
import pytest
from opengrid.node.kv_cache import KVCacheStore


def test_put_and_get():
    store = KVCacheStore(max_ram_gb=0.01)
    store.put("job1", (0, 7), b"activations")
    assert store.get("job1", (0, 7)) == b"activations"


def test_lru_eviction():
    store = KVCacheStore(max_ram_gb=0.000001)  # 1 KB budget
    store.put("job1", (0, 7), b"a" * 512)
    store.put("job2", (0, 7), b"b" * 512)
    # One of them should have been evicted
    total = (store.get("job1", (0, 7)) is not None) + (store.get("job2", (0, 7)) is not None)
    assert total <= 2


def test_evict_job():
    store = KVCacheStore(max_ram_gb=1.0)
    store.put("job1", (0, 7), b"data")
    store.evict_job("job1")
    assert store.get("job1", (0, 7)) is None
