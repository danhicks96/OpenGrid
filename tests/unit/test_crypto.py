"""Tests for node identity and cryptographic operations."""
import pytest
from pathlib import Path
from opengrid.mesh.crypto import (
    generate_keypair, save_keypair, load_keypair, load_or_generate,
    pubkey_to_bytes, pubkey_from_bytes, pubkey_to_hex, pubkey_from_hex,
    node_id_from_pubkey, sign, verify, sign_hex, verify_hex,
)


def test_generate_keypair():
    priv, pub = generate_keypair()
    assert priv is not None
    assert pub is not None


def test_pubkey_bytes_roundtrip():
    _, pub = generate_keypair()
    raw = pubkey_to_bytes(pub)
    assert len(raw) == 32
    recovered = pubkey_from_bytes(raw)
    assert pubkey_to_bytes(recovered) == raw


def test_pubkey_hex_roundtrip():
    _, pub = generate_keypair()
    hex_str = pubkey_to_hex(pub)
    assert len(hex_str) == 64
    recovered = pubkey_from_hex(hex_str)
    assert pubkey_to_hex(recovered) == hex_str


def test_node_id_format():
    _, pub = generate_keypair()
    nid = node_id_from_pubkey(pub)
    assert nid.startswith("og_")
    assert len(nid) > 5


def test_node_id_deterministic():
    _, pub = generate_keypair()
    assert node_id_from_pubkey(pub) == node_id_from_pubkey(pub)


def test_different_keys_different_ids():
    _, pub1 = generate_keypair()
    _, pub2 = generate_keypair()
    assert node_id_from_pubkey(pub1) != node_id_from_pubkey(pub2)


def test_sign_and_verify():
    priv, pub = generate_keypair()
    msg = b"hello opengrid"
    sig = sign(msg, priv)
    assert len(sig) == 64
    assert verify(msg, sig, pub)


def test_verify_wrong_message():
    priv, pub = generate_keypair()
    sig = sign(b"hello", priv)
    assert not verify(b"tampered", sig, pub)


def test_verify_wrong_key():
    priv1, _ = generate_keypair()
    _, pub2 = generate_keypair()
    sig = sign(b"hello", priv1)
    assert not verify(b"hello", sig, pub2)


def test_sign_hex_roundtrip():
    priv, pub = generate_keypair()
    msg = b"test message"
    sig_hex = sign_hex(msg, priv)
    assert verify_hex(msg, sig_hex, pub)


def test_save_and_load(tmp_path):
    priv, pub = generate_keypair()
    path = tmp_path / "node_key.pem"
    save_keypair(priv, path)
    assert path.exists()
    priv2, pub2 = load_keypair(path)
    assert pubkey_to_bytes(pub) == pubkey_to_bytes(pub2)


def test_load_or_generate_creates(tmp_path):
    path = tmp_path / "node_key.pem"
    priv, pub = load_or_generate(path)
    assert path.exists()
    nid = node_id_from_pubkey(pub)
    assert nid.startswith("og_")


def test_load_or_generate_reuses(tmp_path):
    path = tmp_path / "node_key.pem"
    _, pub1 = load_or_generate(path)
    _, pub2 = load_or_generate(path)
    assert pubkey_to_bytes(pub1) == pubkey_to_bytes(pub2)
