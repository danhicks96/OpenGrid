"""
Node identity and cryptographic operations.

Every OpenGrid node is identified by an Ed25519 public key — never by IP.
The pubkey is generated once on first run and saved to ~/.opengrid/node_key.pem.
The node_id is a base58-encoded hash of the pubkey, prefixed with 'og_'.

No IP address is ever stored, logged, or transmitted as part of node identity.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

log = logging.getLogger(__name__)

# Base58 alphabet (Bitcoin-style, no 0/O/I/l ambiguity)
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58_encode(data: bytes) -> str:
    """Encode bytes to base58 string."""
    n = int.from_bytes(data, "big")
    result = []
    while n > 0:
        n, r = divmod(n, 58)
        result.append(_B58_ALPHABET[r:r+1])
    # Preserve leading zeros
    for byte in data:
        if byte == 0:
            result.append(_B58_ALPHABET[0:1])
        else:
            break
    return b"".join(reversed(result)).decode()


def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a new Ed25519 keypair."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def save_keypair(private_key: Ed25519PrivateKey, path: Path) -> None:
    """Save private key to PEM file (public key is derived from it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # Write with restrictive permissions
    path.write_bytes(pem)
    os.chmod(path, 0o600)
    log.info("Node keypair saved to %s", path)


def load_keypair(path: Path) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Load keypair from PEM file."""
    pem = path.read_bytes()
    private_key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError(f"Expected Ed25519 key, got {type(private_key)}")
    return private_key, private_key.public_key()


def load_or_generate(path: Path) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Load existing keypair or generate a new one."""
    if path.exists():
        log.debug("Loading existing keypair from %s", path)
        return load_keypair(path)
    log.info("Generating new node keypair...")
    private_key, public_key = generate_keypair()
    save_keypair(private_key, path)
    return private_key, public_key


def pubkey_to_bytes(public_key: Ed25519PublicKey) -> bytes:
    """Get the raw 32 bytes of a public key."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def pubkey_from_bytes(data: bytes) -> Ed25519PublicKey:
    """Reconstruct a public key from 32 raw bytes."""
    return Ed25519PublicKey.from_public_bytes(data)


def node_id_from_pubkey(public_key: Ed25519PublicKey) -> str:
    """
    Derive a human-readable node ID from a public key.
    Format: og_<base58(sha256(pubkey_bytes)[:20])>
    Example: og_7Kx9mPdR4nVqW2yZ
    """
    raw = pubkey_to_bytes(public_key)
    digest = hashlib.sha256(raw).digest()[:20]  # 160 bits, same as Bitcoin addresses
    return "og_" + _base58_encode(digest)


def pubkey_to_hex(public_key: Ed25519PublicKey) -> str:
    """Hex-encode a public key for transmission."""
    return pubkey_to_bytes(public_key).hex()


def pubkey_from_hex(hex_str: str) -> Ed25519PublicKey:
    """Reconstruct a public key from hex string."""
    return pubkey_from_bytes(bytes.fromhex(hex_str))


def sign(message: bytes, private_key: Ed25519PrivateKey) -> bytes:
    """Sign a message with the node's private key. Returns 64-byte signature."""
    return private_key.sign(message)


def verify(message: bytes, signature: bytes, public_key: Ed25519PublicKey) -> bool:
    """Verify a signature against a public key. Returns True if valid."""
    try:
        public_key.verify(signature, message)
        return True
    except Exception:
        return False


def sign_hex(message: bytes, private_key: Ed25519PrivateKey) -> str:
    """Sign and return hex-encoded signature."""
    return sign(message, private_key).hex()


def verify_hex(message: bytes, signature_hex: str, public_key: Ed25519PublicKey) -> bool:
    """Verify a hex-encoded signature."""
    try:
        return verify(message, bytes.fromhex(signature_hex), public_key)
    except ValueError:
        return False
