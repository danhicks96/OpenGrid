"""
Shard cache manager — downloads, verifies (SHA-256), and evicts model shards.
Shards are stored as immutable content-addressed files under ~/.opengrid/shards/.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MB read/write chunks


@dataclass
class ShardMeta:
    model_id: str
    shard_id: int
    layers: list[int]   # [first, last] inclusive
    size_gb: float
    sha256: str
    hf_url: str


@dataclass
class ModelManifest:
    model_id: str
    base_model: str
    quantization: str
    total_layers: int
    shards: list[ShardMeta]

    @classmethod
    def from_dict(cls, d: dict) -> "ModelManifest":
        shards = [ShardMeta(**s) for s in d["shards"]]
        return cls(
            model_id=d["model_id"],
            base_model=d["base_model"],
            quantization=d["quantization"],
            total_layers=d["total_layers"],
            shards=shards,
        )

    @classmethod
    def from_file(cls, path: Path) -> "ModelManifest":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _shard_path(shards_dir: Path, model_id: str, shard_id: int) -> Path:
    return shards_dir / model_id / f"shard-{shard_id:03d}.safetensors"


def shard_is_valid(shards_dir: Path, meta: ShardMeta) -> bool:
    path = _shard_path(shards_dir, meta.model_id, meta.shard_id)
    if not path.exists():
        return False
    return _sha256_file(path) == meta.sha256


def download_shard(shards_dir: Path, meta: ShardMeta, max_disk_gb: float = 40.0) -> Path:
    """Download a shard from HuggingFace if not already cached."""
    dest = _shard_path(shards_dir, meta.model_id, meta.shard_id)
    if shard_is_valid(shards_dir, meta):
        log.info("Shard %s/%d already valid.", meta.model_id, meta.shard_id)
        return dest

    # Check disk budget
    used_gb = _used_disk_gb(shards_dir)
    if used_gb + meta.size_gb > max_disk_gb:
        evict_lru(shards_dir, needed_gb=meta.size_gb, budget_gb=max_disk_gb)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    log.info("Downloading shard %s/%d from %s", meta.model_id, meta.shard_id, meta.hf_url)
    try:
        urllib.request.urlretrieve(meta.hf_url, tmp)
        actual = _sha256_file(tmp)
        if actual != meta.sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for shard {meta.shard_id}: expected {meta.sha256}, got {actual}"
            )
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _used_disk_gb(shards_dir: Path) -> float:
    if not shards_dir.exists():
        return 0.0
    total = sum(p.stat().st_size for p in shards_dir.rglob("*.safetensors") if p.is_file())
    return total / 1e9


def evict_lru(shards_dir: Path, needed_gb: float, budget_gb: float) -> None:
    """Remove least-recently-used shards until needed_gb fits within budget_gb."""
    shards = sorted(
        shards_dir.rglob("*.safetensors"),
        key=lambda p: p.stat().st_atime,
    )
    freed = 0.0
    current_used = _used_disk_gb(shards_dir)
    for path in shards:
        if current_used - freed + needed_gb <= budget_gb:
            break
        size = path.stat().st_size / 1e9
        log.info("Evicting shard %s (%.2f GB)", path.name, size)
        path.unlink()
        freed += size
    # Remove empty model dirs
    for d in shards_dir.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


class ShardManager:
    def __init__(self, shards_dir: Path, max_disk_gb: float = 40.0):
        self.shards_dir = shards_dir
        self.max_disk_gb = max_disk_gb
        shards_dir.mkdir(parents=True, exist_ok=True)

    def get(self, manifest: ModelManifest, shard_id: int) -> Path:
        meta = manifest.shards[shard_id]
        return download_shard(self.shards_dir, meta, self.max_disk_gb)

    def has(self, manifest: ModelManifest, shard_id: int) -> bool:
        return shard_is_valid(self.shards_dir, manifest.shards[shard_id])

    def path(self, model_id: str, shard_id: int) -> Path:
        return _shard_path(self.shards_dir, model_id, shard_id)
