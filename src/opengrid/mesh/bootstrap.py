"""Bootstrap node list and DNS seed resolution."""
from __future__ import annotations

import logging
import socket
from typing import Optional

log = logging.getLogger(__name__)

# Hardcoded seed nodes — updated before each public release
HARDCODED_SEEDS: list[tuple[str, int]] = [
    ("bootstrap1.opengrid.network", 7600),
    ("bootstrap2.opengrid.network", 7600),
    ("bootstrap3.opengrid.network", 7600),
]

DNS_SEED_HOSTNAME = "seed.opengrid.network"


def resolve_bootstrap(extra: list[tuple[str, int]] | None = None) -> list[tuple[str, int]]:
    """
    Returns a deduplicated list of bootstrap addresses by combining:
    1. Hardcoded seeds
    2. DNS seed resolution
    3. Any caller-supplied extras
    """
    result: list[tuple[str, int]] = list(HARDCODED_SEEDS)

    try:
        infos = socket.getaddrinfo(DNS_SEED_HOSTNAME, 7600, proto=socket.IPPROTO_TCP)
        for info in infos:
            ip = info[4][0]
            result.append((ip, 7600))
    except socket.gaierror:
        log.debug("DNS seed %s not reachable (expected in dev)", DNS_SEED_HOSTNAME)

    if extra:
        result.extend(extra)

    # Deduplicate
    seen: set[tuple[str, int]] = set()
    unique = []
    for addr in result:
        if addr not in seen:
            seen.add(addr)
            unique.append(addr)
    return unique
