from __future__ import annotations

import os
from pathlib import Path

import pytest
from local_meetscribe.security import GeminiShareStore
from local_meetscribe.utils.errors import LocalMeetScribeError


def test_share_store_rejects_short_internet_passcodes(tmp_path: Path) -> None:
    store = GeminiShareStore(tmp_path, protect=lambda value: value, unprotect=lambda value: value)

    with pytest.raises(LocalMeetScribeError, match="at least 8"):
        store.configure_passcode("1111")


def test_share_store_hashes_passcode_and_does_not_write_raw_key(tmp_path: Path) -> None:
    store = GeminiShareStore(
        tmp_path,
        protect=lambda value: f"protected:{value[::-1]}",
        unprotect=lambda value: value.removeprefix("protected:")[::-1],
    )

    store.configure_passcode("98769876")
    store.save_api_key("test-gemini-api-key-1234567890")

    contents = store.path.read_text(encoding="utf-8")
    assert store.passcode_configured is True
    assert store.api_key_configured is True
    assert store.verify_passcode("98769876") is True
    assert store.verify_passcode("1111") is False
    assert store.load_api_key() == "test-gemini-api-key-1234567890"
    assert "98769876" not in contents
    assert "test-gemini-api-key-1234567890" not in contents


def test_share_store_configures_hashed_mode_access_codes_and_replaces_legacy(
    tmp_path: Path,
) -> None:
    store = GeminiShareStore(
        tmp_path,
        protect=lambda value: value,
        unprotect=lambda value: value,
    )
    store.configure_passcode("legacy-record-code")
    store.configure_access_codes(record_code="1234", upload_code="5678")

    contents = store.path.read_text(encoding="utf-8")
    assert store.passcode_configured is True
    assert store.access_codes_configured is True
    assert store.access_mode_for_passcode("1234") == "record"
    assert store.access_mode_for_passcode("5678") == "upload"
    assert store.access_mode_for_passcode("legacy-record-code") is None
    assert store.access_mode_for_passcode("9999") is None
    assert "1234" not in contents
    assert "5678" not in contents
    assert "legacy-record-code" not in contents


def test_share_store_switching_back_to_one_passcode_removes_mode_codes(tmp_path: Path) -> None:
    store = GeminiShareStore(tmp_path, protect=lambda value: value, unprotect=lambda value: value)
    store.configure_access_codes(record_code="1234", upload_code="5678")

    store.configure_passcode("replacement-record-passcode")

    assert store.access_codes_configured is False
    assert store.access_mode_for_passcode("1234") is None
    assert store.access_mode_for_passcode("5678") is None
    assert store.access_mode_for_passcode("replacement-record-passcode") == "record"


def test_saving_api_key_preserves_access_code_config_version(tmp_path: Path) -> None:
    store = GeminiShareStore(
        tmp_path,
        protect=lambda value: f"protected:{value}",
        unprotect=lambda value: value.removeprefix("protected:"),
    )
    store.configure_access_codes(record_code="1234", upload_code="5678")

    store.save_api_key("test-gemini-api-key-1234567890")

    payload = store._read()  # noqa: SLF001
    assert payload["version"] == 2
    assert store.access_mode_for_passcode("1234") == "record"
    assert store.access_mode_for_passcode("5678") == "upload"


@pytest.mark.parametrize(
    ("record_code", "upload_code", "message"),
    [
        ("123", "5678", "at least 4 digits"),
        ("1234", "abcd", "at least 4 digits"),
        ("1234", "1234", "must be different"),
    ],
)
def test_share_store_rejects_invalid_mode_access_codes(
    tmp_path: Path,
    record_code: str,
    upload_code: str,
    message: str,
) -> None:
    store = GeminiShareStore(tmp_path, protect=lambda value: value, unprotect=lambda value: value)

    with pytest.raises(LocalMeetScribeError, match=message):
        store.configure_access_codes(record_code=record_code, upload_code=upload_code)

    assert store.passcode_configured is False


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI integration test")
def test_windows_dpapi_share_key_round_trip(tmp_path: Path) -> None:
    store = GeminiShareStore(tmp_path)

    store.configure_passcode("98769876")
    store.save_api_key("test-gemini-api-key-1234567890")

    assert store.load_api_key() == "test-gemini-api-key-1234567890"
    assert "test-gemini-api-key-1234567890" not in store.path.read_text(encoding="utf-8")
