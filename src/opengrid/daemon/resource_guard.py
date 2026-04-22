"""
Resource guard — enforces the limits from config.toml at runtime.
Checks battery state, GPU load (gaming detection), and active-hours schedule.
"""
from __future__ import annotations

import datetime
import logging
import subprocess
from typing import Optional

from opengrid.daemon.config import Config

log = logging.getLogger(__name__)


def _is_on_battery() -> bool:
    try:
        import psutil
        bat = psutil.sensors_battery()
        return bat is not None and not bat.power_plugged
    except Exception:
        pass
    # Fallback: /sys/class/power_supply
    try:
        for path in [
            "/sys/class/power_supply/AC/online",
            "/sys/class/power_supply/ACAD/online",
        ]:
            try:
                with open(path) as f:
                    return f.read().strip() == "0"
            except OSError:
                continue
    except Exception:
        pass
    return False


def _gpu_load_percent() -> float:
    """Return GPU utilization 0–100.  Returns 0 if GPU not found."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip().splitlines()[0]
        return float(out)
    except Exception:
        return 0.0


def _parse_hhmm(s: str) -> datetime.time:
    h, m = s.split(":")
    return datetime.time(int(h), int(m))


class ResourceGuard:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def should_pause(self) -> tuple[bool, str]:
        """
        Returns (True, reason) if the node should pause serving jobs right now.
        """
        c = self.cfg

        if c.schedule.pause_on_battery and _is_on_battery():
            return True, "on battery"

        if c.schedule.pause_when_gaming:
            load = _gpu_load_percent()
            if load > 80:
                return True, f"GPU busy ({load:.0f}% — gaming/other process)"

        now = datetime.datetime.now().time()
        start = _parse_hhmm(c.schedule.active_hours_start)
        end = _parse_hhmm(c.schedule.active_hours_end)
        if start < end:
            in_window = start <= now <= end
        else:  # overnight window, e.g. 22:00–06:00
            in_window = now >= start or now <= end
        if not in_window:
            return True, f"outside active hours ({c.schedule.active_hours_start}–{c.schedule.active_hours_end})"

        return False, ""

    def gpu_budget_gb(self) -> float:
        return self.cfg.resources.max_vram_gb

    def ram_budget_gb(self) -> float:
        return self.cfg.resources.max_ram_gb

    def disk_budget_gb(self) -> float:
        return self.cfg.resources.max_disk_gb
