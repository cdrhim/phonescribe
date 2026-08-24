from __future__ import annotations

import os
from pathlib import Path

import pytest
from local_meetscribe.security import GeminiShareStore


def test_share_store_hashes_passcode_and_does_not_write_raw_key(tmp_path: Path) -> None:
    store = GeminiShareStore(
        tmp_path,
        protect=lambda value: f"protected:{value[::-1]}",
        unprotect=lambda value: value.removeprefix("protected:")[::-1],
    )

    store.configure_passcode("3543")
    store.save_api_key("test-gemini-api-key-1234567890")

    contents = store.path.read_text(encoding="utf-8")
    assert store.passcode_configured is True
    assert store.api_key_configured is True
    assert store.verify_passcode("3543") is True
    assert store.verify_passcode("0000") is False
    assert store.load_api_key() == "test-gemini-api-key-1234567890"
    assert "3543" not in contents
    assert "test-gemini-api-key-1234567890" not in contents


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI integration test")
def test_windows_dpapi_share_key_round_trip(tmp_path: Path) -> None:
    store = GeminiShareStore(tmp_path)

    store.configure_passcode("3543")
    store.save_api_key("test-gemini-api-key-1234567890")

    assert store.load_api_key() == "test-gemini-api-key-1234567890"
    assert "test-gemini-api-key-1234567890" not in store.path.read_text(encoding="utf-8")
