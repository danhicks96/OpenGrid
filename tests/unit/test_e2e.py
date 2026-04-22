"""Tests for end-to-end encryption between nodes."""
import pytest
from nacl.exceptions import CryptoError
from opengrid.mesh.crypto import generate_keypair, pubkey_to_bytes
from opengrid.mesh.e2e import E2ECipher, SealedCipher, encrypt_for, decrypt_from
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption


def _get_raw_private(priv) -> bytes:
    """Extract raw 32-byte seed from Ed25519PrivateKey."""
    raw = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    return raw


def test_e2e_encrypt_decrypt():
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    raw_a = _get_raw_private(priv_a)
    raw_b = _get_raw_private(priv_b)
    pub_a_bytes = pubkey_to_bytes(pub_a)
    pub_b_bytes = pubkey_to_bytes(pub_b)

    # A encrypts for B
    cipher_a = E2ECipher(raw_a, pub_b_bytes)
    encrypted = cipher_a.encrypt(b"hello from A")

    # B decrypts from A
    cipher_b = E2ECipher(raw_b, pub_a_bytes)
    plaintext = cipher_b.decrypt(encrypted)
    assert plaintext == b"hello from A"


def test_e2e_tampered_message():
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    raw_a = _get_raw_private(priv_a)
    raw_b = _get_raw_private(priv_b)
    pub_a_bytes = pubkey_to_bytes(pub_a)
    pub_b_bytes = pubkey_to_bytes(pub_b)

    cipher_a = E2ECipher(raw_a, pub_b_bytes)
    encrypted = cipher_a.encrypt(b"original")

    # Tamper with the ciphertext
    tampered = bytearray(encrypted)
    tampered[-1] ^= 0xFF
    tampered = bytes(tampered)

    cipher_b = E2ECipher(raw_b, pub_a_bytes)
    with pytest.raises(CryptoError):
        cipher_b.decrypt(tampered)


def test_e2e_wrong_recipient():
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    priv_c, pub_c = generate_keypair()
    raw_a = _get_raw_private(priv_a)
    raw_c = _get_raw_private(priv_c)
    pub_a_bytes = pubkey_to_bytes(pub_a)
    pub_b_bytes = pubkey_to_bytes(pub_b)

    # A encrypts for B
    cipher_a = E2ECipher(raw_a, pub_b_bytes)
    encrypted = cipher_a.encrypt(b"secret")

    # C tries to decrypt (wrong key)
    cipher_c = E2ECipher(raw_c, pub_a_bytes)
    with pytest.raises(CryptoError):
        cipher_c.decrypt(encrypted)


def test_sealed_encrypt_decrypt():
    priv_b, pub_b = generate_keypair()
    raw_b = _get_raw_private(priv_b)
    pub_b_bytes = pubkey_to_bytes(pub_b)

    # Anonymous sender encrypts for B
    encrypted = SealedCipher.encrypt(b"anonymous message", pub_b_bytes)

    # B decrypts
    plaintext = SealedCipher.decrypt(encrypted, raw_b)
    assert plaintext == b"anonymous message"


def test_convenience_functions():
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    raw_a = _get_raw_private(priv_a)
    raw_b = _get_raw_private(priv_b)
    pub_a_bytes = pubkey_to_bytes(pub_a)
    pub_b_bytes = pubkey_to_bytes(pub_b)

    encrypted = encrypt_for(b"hello", raw_a, pub_b_bytes)
    plaintext = decrypt_from(encrypted, raw_b, pub_a_bytes)
    assert plaintext == b"hello"


def test_large_payload():
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    raw_a = _get_raw_private(priv_a)
    raw_b = _get_raw_private(priv_b)
    pub_b_bytes = pubkey_to_bytes(pub_b)
    pub_a_bytes = pubkey_to_bytes(pub_a)

    # Simulate activation tensor (1MB)
    payload = b"\x42" * (1024 * 1024)
    encrypted = encrypt_for(payload, raw_a, pub_b_bytes)
    plaintext = decrypt_from(encrypted, raw_b, pub_a_bytes)
    assert plaintext == payload
