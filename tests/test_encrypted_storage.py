"""Tests for the encrypted storage modules.

This test will prefer the SQLAlchemy implementation when available. If
SQLAlchemy isn't installed the test will be skipped.
"""

import importlib
import sys

import pytest


def try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def test_sqlalchemy_storage_roundtrip():
    storage = try_import("backend.fastapi.security.storage_sqlalchemy")
    if storage is None:
        pytest.skip("SQLAlchemy not installed; skipping SQLAlchemy storage test")

    # Initialize DB and do a simple roundtrip
    storage.init_db()
    rowid = storage.store_encrypted_summary(
        device_id="test-device",
        algorithm="AES-GCM",
        wrapped_dek_hex="deadbeef",
        ciphertext_hex="cafebabe",
        metadata={"note": "unit-test"},
    )
    assert isinstance(rowid, int) and rowid > 0

    rows = storage.list_encrypted_summaries(limit=10)
    assert any(r.get("device_id") == "test-device" for r in rows)
