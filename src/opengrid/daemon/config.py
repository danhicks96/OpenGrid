"""
Daemon configuration — loads and validates ~/.opengrid/config.toml.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-reuse-imports]

OPENGRID_HOME = Path(os.environ.get("OPENGRID_HOME", Path.home() / ".opengrid"))

_DEFAULTS = {
    "resources": {
        "max_gpu_fraction": 0.50,
        "max_vram_gb": 6.0,
        "max_ram_gb": 4.0,
        "max_disk_gb": 40.0,
        "max_upload_mbps": 20.0,
        "max_cpu_fraction": 0.25,
    },
    "schedule": {
        "active_hours_start": "08:00",
        "active_hours_end": "17:00",
        "pause_on_battery": True,
        "pause_when_gaming": True,
    },
    "jobs": {
        "allowed_types": ["inference", "validation"],
        "max_sequence_length": 4096,
    },
    "network": {
        "listen_port": 7600,
        "coordinator_port": 7700,
        "api_port": 8080,
    },
}


@dataclass
class ResourceConfig:
    max_gpu_fraction: float = 0.50
    max_vram_gb: float = 6.0
    max_ram_gb: float = 4.0
    max_disk_gb: float = 40.0
    max_upload_mbps: float = 20.0
    max_cpu_fraction: float = 0.25


@dataclass
class ScheduleConfig:
    active_hours_start: str = "08:00"
    active_hours_end: str = "17:00"
    pause_on_battery: bool = True
    pause_when_gaming: bool = True


@dataclass
class JobsConfig:
    allowed_types: list[str] = field(default_factory=lambda: ["inference", "validation"])
    max_sequence_length: int = 4096


@dataclass
class NetworkConfig:
    listen_port: int = 7600
    coordinator_port: int = 7700
    api_port: int = 8080


@dataclass
class Config:
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    home: Path = field(default_factory=lambda: OPENGRID_HOME)

    @property
    def config_path(self) -> Path:
        return self.home / "config.toml"

    @property
    def profile_path(self) -> Path:
        return self.home / "profile.json"

    @property
    def shards_dir(self) -> Path:
        return self.home / "shards"

    @property
    def ledger_path(self) -> Path:
        return self.home / "ledger.db"


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file, falling back to defaults for missing keys."""
    config_path = path or (OPENGRID_HOME / "config.toml")
    raw: dict = dict(_DEFAULTS)
    if config_path.exists():
        with open(config_path, "rb") as f:
            file_data = tomllib.load(f)
        raw = _deep_merge(raw, file_data)

    return Config(
        resources=ResourceConfig(**raw["resources"]),
        schedule=ScheduleConfig(**raw["schedule"]),
        jobs=JobsConfig(**raw["jobs"]),
        network=NetworkConfig(**raw["network"]),
        home=config_path.parent,
    )


def ensure_dirs(cfg: Config) -> None:
    """Create required directories under OPENGRID_HOME."""
    for d in [cfg.home, cfg.shards_dir]:
        d.mkdir(parents=True, exist_ok=True)
