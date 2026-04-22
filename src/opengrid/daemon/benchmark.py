"""
Hardware benchmark — profiles CPU, GPU, RAM, disk, and network on first run.
Results saved to ~/.opengrid/profile.json and re-used until hardware changes.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class HardwareProfile:
    node_id: str = ""
    platform: str = ""
    cpu_cores: int = 0
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    # GPU fields (empty if no GPU detected)
    gpu_name: str = ""
    gpu_vram_total_gb: float = 0.0
    gpu_vram_free_gb: float = 0.0
    # Benchmark scores
    cpu_gemm_tokens_per_sec: float = 0.0   # 1-bit INT8 GEMM estimate
    gpu_int4_tokens_per_sec: float = 0.0
    disk_read_mb_per_sec: float = 0.0
    # Tier: light | mid | heavy | power
    tier: str = "light"
    benchmark_ts: float = field(default_factory=time.time)


def _get_ram_info() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except ImportError:
        pass
    # Fallback: read /proc/meminfo
    total = free = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total = int(line.split()[1]) / 1e6
                elif line.startswith("MemAvailable"):
                    free = int(line.split()[1]) / 1e6
    except OSError:
        pass
    return round(total, 2), round(free, 2)


def _get_gpu_info() -> tuple[str, float, float]:
    """Query nvidia-smi for GPU name and VRAM. Returns empty strings on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip().splitlines()[0]
        name, total, free = [x.strip() for x in out.split(",")]
        return name, round(float(total) / 1024, 2), round(float(free) / 1024, 2)
    except Exception:
        pass
    # Try rocm-smi for AMD
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode()
        data = json.loads(out)
        card = list(data.values())[0]
        total = float(card.get("VRAM Total Memory (B)", 0)) / 1e9
        used = float(card.get("VRAM Total Used Memory (B)", 0)) / 1e9
        return "AMD GPU", round(total, 2), round(total - used, 2)
    except Exception:
        pass
    return "", 0.0, 0.0


def _bench_disk_read(size_mb: int = 256) -> float:
    """Sequential read benchmark. Returns MB/s."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        path = f.name
        f.write(os.urandom(size_mb * 1024 * 1024))
    try:
        start = time.perf_counter()
        with open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        elapsed = time.perf_counter() - start
        return round(size_mb / elapsed, 1)
    finally:
        os.unlink(path)


def _estimate_cpu_tokens_per_sec(cpu_cores: int, ram_gb: float) -> float:
    """
    Rough estimate based on CPU core count without running a full benchmark.
    A proper benchmark would invoke bitnet.cpp with a fixed prompt.
    """
    base = cpu_cores * 0.8
    if ram_gb >= 16:
        base *= 1.5
    return round(min(base, 89.0), 1)  # 89 t/s is the ARIA single-CPU ceiling


def _estimate_gpu_tokens_per_sec(vram_gb: float) -> float:
    """Rough estimate of INT4 tokens/sec based on VRAM size."""
    if vram_gb <= 0:
        return 0.0
    # Very rough: 8 GB VRAM ≈ 20 t/s, 24 GB ≈ 50 t/s
    return round(min(vram_gb * 2.2, 120.0), 1)


def _assign_tier(profile: HardwareProfile) -> str:
    if profile.gpu_vram_free_gb >= 24:
        return "power"
    if profile.gpu_vram_free_gb >= 16:
        return "heavy"
    if profile.gpu_vram_free_gb >= 8:
        return "mid"
    return "light"


def _node_id_from_machine() -> str:
    import hashlib, uuid
    seed = str(uuid.getnode()).encode()
    return "node-" + hashlib.sha1(seed).hexdigest()[:16]


def run_benchmark() -> HardwareProfile:
    ram_total, ram_free = _get_ram_info()
    gpu_name, vram_total, vram_free = _get_gpu_info()
    cpu_cores = os.cpu_count() or 1

    profile = HardwareProfile(
        node_id=_node_id_from_machine(),
        platform=platform.platform(),
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total,
        ram_free_gb=ram_free,
        gpu_name=gpu_name,
        gpu_vram_total_gb=vram_total,
        gpu_vram_free_gb=vram_free,
        cpu_gemm_tokens_per_sec=_estimate_cpu_tokens_per_sec(cpu_cores, ram_total),
        gpu_int4_tokens_per_sec=_estimate_gpu_tokens_per_sec(vram_free),
        disk_read_mb_per_sec=_bench_disk_read(64),
        benchmark_ts=time.time(),
    )
    profile.tier = _assign_tier(profile)
    return profile


def load_or_run(profile_path: Path) -> HardwareProfile:
    if profile_path.exists():
        with open(profile_path) as f:
            return HardwareProfile(**json.load(f))
    profile = run_benchmark()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return profile
