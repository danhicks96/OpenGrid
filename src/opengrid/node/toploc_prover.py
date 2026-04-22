"""
TOPLOC proof stub — generates and verifies locality-sensitive hash proofs
of intermediate activations for trustless inference verification.

Full TOPLOC implementation: https://github.com/PrimeIntellect-ai/toploc
This module provides the interface; connect to the real toploc library
when available, otherwise uses a placeholder hash.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import struct
from typing import Optional

log = logging.getLogger(__name__)

PROOF_BYTES_PER_32_TOKENS = 258  # per TOPLOC paper


def _stub_proof(activations: bytes, top_k: int = 16) -> bytes:
    """
    Placeholder: SHA-256 hash of the top-k bytes of activations.
    Replace with real TOPLOC polynomial encoding when the library is available.
    """
    h = hashlib.sha256(activations[:top_k * 4]).digest()
    return h[:PROOF_BYTES_PER_32_TOKENS // 2]  # truncate to realistic size


def generate_proof(activations: bytes, token_count: int = 32) -> str:
    """
    Generate a TOPLOC proof for a block of activations.
    Returns base64-encoded proof string.
    """
    try:
        import toploc  # type: ignore
        proof_bytes = toploc.generate(activations, token_count)
    except ImportError:
        proof_bytes = _stub_proof(activations)

    return base64.b64encode(proof_bytes).decode()


def verify_proof(activations: bytes, proof_b64: str, token_count: int = 32) -> bool:
    """
    Verify a TOPLOC proof against activations.
    Returns True if valid, False if tampered.
    """
    try:
        import toploc  # type: ignore
        proof_bytes = base64.b64decode(proof_b64)
        return toploc.verify(activations, proof_bytes, token_count)
    except ImportError:
        # Stub: regenerate and compare
        expected = _stub_proof(activations)
        try:
            received = base64.b64decode(proof_b64)
            return received == expected
        except Exception:
            return False
    except Exception as e:
        log.warning("TOPLOC verify error: %s", e)
        return False
