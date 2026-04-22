"""
Model registry — loads and exposes model manifests from the manifests/ directory.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from opengrid.daemon.shard_manager import ModelManifest

log = logging.getLogger(__name__)

_MANIFESTS_DIR = Path(__file__).parent / "manifests"


class ModelRegistry:
    def __init__(self, manifests_dir: Path | None = None):
        self._dir = manifests_dir or _MANIFESTS_DIR
        self._models: dict[str, ModelManifest] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self._dir.exists():
            log.warning("Manifests directory not found: %s", self._dir)
            return
        for path in self._dir.glob("*.json"):
            try:
                manifest = ModelManifest.from_file(path)
                self._models[manifest.model_id] = manifest
                log.debug("Loaded manifest: %s", manifest.model_id)
            except Exception as e:
                log.warning("Failed to load manifest %s: %s", path.name, e)

    def get(self, model_id: str) -> Optional[ModelManifest]:
        return self._models.get(model_id)

    def list_models(self) -> list[str]:
        return sorted(self._models.keys())

    def register(self, manifest: ModelManifest) -> None:
        self._models[manifest.model_id] = manifest
