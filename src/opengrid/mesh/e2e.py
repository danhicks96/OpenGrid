"""
End-to-end encryption between OpenGrid nodes.

Uses NaCl Box (Curve25519 key agreement + XSalsa20-Poly1305 authenticated encryption).
All activation payloads and work packets are encrypted before transmission —
relay nodes see only opaque bytes.

Ed25519 signing keys are converted to Curve25519 for encryption (standard practice,
used by Signal, WireGuard, etc.).
"""
from __future__ import annotations

import logging
from typing import Optional

import nacl.utils
from nacl.public import Box, PrivateKey, PublicKey, SealedBox
from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import CryptoError

log = logging.getLogger(__name__)


def ed25519_to_curve25519_private(signing_key_bytes: bytes) -> PrivateKey:
    """Convert Ed25519 private key to Curve25519 for encryption."""
    sk = SigningKey(signing_key_bytes)
    return sk.to_curve25519_private_key()


def ed25519_to_curve25519_public(verify_key_bytes: bytes) -> PublicKey:
    """Convert Ed25519 public key to Curve25519 for encryption."""
    vk = VerifyKey(verify_key_bytes)
    return vk.to_curve25519_public_key()


class E2ECipher:
    """
    End-to-end encryption between two nodes.

    Usage:
        # Node A encrypts for Node B
        cipher_a = E2ECipher(my_private_key_bytes, peer_public_key_bytes)
        encrypted = cipher_a.encrypt(b"hello from A")

        # Node B decrypts from Node A
        cipher_b = E2ECipher(my_private_key_bytes, peer_public_key_bytes)
        plaintext = cipher_b.decrypt(encrypted)
    """

    def __init__(self, my_ed25519_private: bytes, peer_ed25519_public: bytes):
        my_curve = ed25519_to_curve25519_private(my_ed25519_private)
        peer_curve = ed25519_to_curve25519_public(peer_ed25519_public)
        self._box = Box(my_curve, peer_curve)

    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt and authenticate a message for the peer.
        Returns nonce + ciphertext (nonce is prepended automatically by PyNaCl).
        The relay cannot read or modify this.
        """
        return self._box.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt and verify a message from the peer.
        Raises CryptoError if tampered or wrong sender.
        """
        return self._box.decrypt(ciphertext)


class SealedCipher:
    """
    Anonymous encryption — encrypt for a pubkey without revealing your identity.
    Used for initial handshake messages where the sender wants to remain anonymous
    even to the recipient until they choose to reveal themselves.
    """

    @staticmethod
    def encrypt(plaintext: bytes, recipient_ed25519_public: bytes) -> bytes:
        """Encrypt anonymously for a recipient. Only they can decrypt."""
        recipient_curve = ed25519_to_curve25519_public(recipient_ed25519_public)
        sealed = SealedBox(recipient_curve)
        return sealed.encrypt(plaintext)

    @staticmethod
    def decrypt(ciphertext: bytes, my_ed25519_private: bytes) -> bytes:
        """Decrypt an anonymously-encrypted message."""
        my_curve = ed25519_to_curve25519_private(my_ed25519_private)
        sealed = SealedBox(my_curve)
        return sealed.decrypt(ciphertext)


def encrypt_for(payload: bytes, my_private: bytes, recipient_public: bytes) -> bytes:
    """Convenience: encrypt payload for a specific recipient."""
    cipher = E2ECipher(my_private, recipient_public)
    return cipher.encrypt(payload)


def decrypt_from(encrypted: bytes, my_private: bytes, sender_public: bytes) -> bytes:
    """Convenience: decrypt payload from a specific sender."""
    cipher = E2ECipher(my_private, sender_public)
    return cipher.decrypt(encrypted)
