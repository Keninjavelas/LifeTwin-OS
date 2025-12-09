"""Simple envelope encryption helpers (POC)

This file provides a small proof-of-concept for envelope encryption using
the `cryptography` package. Production systems should use a vetted KMS/HSM
for KEK storage and key wrapping.

Usage:
  - generate a random DEK for each payload, encrypt payload with AES-GCM
  - wrap the DEK using a KEK (which on Android would be stored in Keystore)
  - store ciphertext + wrapped_dek + metadata on server

Note: This module intentionally keeps things small and explicit for review.
"""

from typing import Tuple, Dict
import os
import json

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.asymmetric import rsa
except Exception:
    # When the environment doesn't have cryptography installed, provide
    # useful error messages instead of failing imports at module load time.
    AESGCM = None


def generate_dek(length: int = 32) -> bytes:
    """Generate a random Data Encryption Key (DEK).

    Args:
        length: key length in bytes (32 = 256-bit)
    Returns:
        random bytes
    """
    return os.urandom(length)


def encrypt_payload(dek: bytes, plaintext: bytes, associated_data: bytes = None) -> Dict:
    """Encrypt plaintext with AES-GCM using the provided DEK.

    Returns a dict with ciphertext, nonce, and tag encoded as base64-like bytes.
    """
    if AESGCM is None:
        raise RuntimeError("cryptography is required for encryption (install `cryptography`)")
    aesgcm = AESGCM(dek)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    return {"nonce": nonce.hex(), "ciphertext": ct.hex()}


def decrypt_payload(dek: bytes, payload: Dict, associated_data: bytes = None) -> bytes:
    """Decrypt AES-GCM encrypted payload produced by `encrypt_payload`.
    """
    if AESGCM is None:
        raise RuntimeError("cryptography is required for decryption (install `cryptography`)")
    aesgcm = AESGCM(dek)
    nonce = bytes.fromhex(payload["nonce"])
    ct = bytes.fromhex(payload["ciphertext"])
    return aesgcm.decrypt(nonce, ct, associated_data)


# Simple RSA-based KEK wrap/unwarp for demo purposes only.
def generate_rsa_kek(key_size: int = 2048) -> Tuple[bytes, bytes]:
    """Generate RSA keypair material (PEM-encoded private and public keys).

    In a real deployment the KEK would be kept in a platform keystore or KMS.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv_pem, pub_pem


def wrap_dek_with_rsa(public_key_pem: bytes, dek: bytes) -> bytes:
    """Wrap (encrypt) DEK using receiver's RSA public key (OAEP).

    Returns wrapped (encrypted) bytes.
    """
    if AESGCM is None:
        raise RuntimeError("cryptography is required for key wrapping")
    public_key = serialization.load_pem_public_key(public_key_pem)
    wrapped = public_key.encrypt(
        dek,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    return wrapped


def unwrap_dek_with_rsa(private_key_pem: bytes, wrapped: bytes) -> bytes:
    """Unwrap DEK using RSA private key (OAEP).
    """
    if AESGCM is None:
        raise RuntimeError("cryptography is required for key unwrapping")
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    dek = private_key.decrypt(
        wrapped,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    return dek


def example_flow() -> None:
    """Demonstrate an example envelope encryption flow (prints intermediate lengths).

    This is a convenience for manual smoke-testing; not used automatically.
    """
    dek = generate_dek()
    print(f"DEK length: {len(dek)}")
    data = b"hello encrypted summaries"
    payload = encrypt_payload(dek, data)
    print("Encrypted payload keys:", list(payload.keys()))
    priv, pub = generate_rsa_kek()
    wrapped = wrap_dek_with_rsa(pub, dek)
    print("Wrapped DEK length:", len(wrapped))
    dek2 = unwrap_dek_with_rsa(priv, wrapped)
    assert dek == dek2
    print("Roundtrip OK")


if __name__ == "__main__":
    try:
        example_flow()
    except Exception as e:
        print("POC run failed; ensure `cryptography` is installed:", e)
