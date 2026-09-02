from __future__ import annotations

import base64
import contextlib
import ctypes
import hashlib
import hmac
import json
import os
import secrets
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from local_meetscribe.utils.errors import LocalMeetScribeError

PBKDF2_ITERATIONS = 240_000


@dataclass(frozen=True)
class SupabaseStoredConfig:
    project_url: str
    service_role_key: str
    bucket: str


class SupabaseConfigStore:
    """DPAPI-backed Supabase service credential owned by the server PC user."""

    def __init__(
        self,
        data_dir: Path,
        *,
        protect: Callable[[str], str] | None = None,
        unprotect: Callable[[str], str] | None = None,
    ) -> None:
        self.path = data_dir / "config" / "supabase.json"
        self._protect = protect or _protect_secret
        self._unprotect = unprotect or _unprotect_secret

    @property
    def configured(self) -> bool:
        payload = self._read()
        return bool(payload.get("project_url") and payload.get("encrypted_service_role_key"))

    def save(
        self,
        *,
        project_url: str,
        service_role_key: str,
        bucket: str = "recordings",
    ) -> None:
        normalized_url = project_url.strip().rstrip("/")
        parsed = urlparse(normalized_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise LocalMeetScribeError("Supabase project URL must be an HTTPS URL.")
        normalized_key = service_role_key.strip()
        if len(normalized_key) < 20:
            raise LocalMeetScribeError("Supabase service role key is missing or too short.")
        normalized_bucket = bucket.strip()
        if normalized_bucket != "recordings":
            raise LocalMeetScribeError("Supabase Storage bucket must be 'recordings'.")
        self._write(
            {
                "version": 1,
                "project_url": normalized_url,
                "encrypted_service_role_key": self._protect(normalized_key),
                "bucket": normalized_bucket,
            }
        )

    def load(self) -> SupabaseStoredConfig | None:
        payload = self._read()
        encrypted = payload.get("encrypted_service_role_key")
        project_url = payload.get("project_url")
        if not isinstance(encrypted, str) or not isinstance(project_url, str):
            return None
        try:
            service_role_key = self._unprotect(encrypted).strip()
        except (OSError, ValueError) as exc:
            raise LocalMeetScribeError(
                "The saved Supabase credential cannot be decrypted by this Windows user."
            ) from exc
        if not service_role_key:
            return None
        bucket = payload.get("bucket")
        return SupabaseStoredConfig(
            project_url=project_url.strip().rstrip("/"),
            service_role_key=service_role_key,
            bucket=str(bucket or "recordings"),
        )

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        with contextlib.suppress(OSError):
            self.path.chmod(0o600)


class GeminiShareStore:
    def __init__(
        self,
        data_dir: Path,
        *,
        protect: Callable[[str], str] | None = None,
        unprotect: Callable[[str], str] | None = None,
    ) -> None:
        self.path = data_dir / "config" / "gemini_share.json"
        self._protect = protect or _protect_secret
        self._unprotect = unprotect or _unprotect_secret

    @property
    def passcode_configured(self) -> bool:
        payload = self._read()
        return bool(payload.get("passcode_salt") and payload.get("passcode_hash"))

    @property
    def api_key_configured(self) -> bool:
        return bool(self._read().get("encrypted_api_key"))

    def configure_passcode(self, passcode: str) -> None:
        normalized = passcode.strip()
        if len(normalized) < 8:
            raise LocalMeetScribeError("Share passcode must contain at least 8 characters.")
        payload = self._read()
        salt = secrets.token_bytes(16)
        payload.update(
            {
                "version": 1,
                "passcode_salt": base64.b64encode(salt).decode("ascii"),
                "passcode_hash": base64.b64encode(_passcode_hash(normalized, salt)).decode("ascii"),
            }
        )
        self._write(payload)

    def verify_passcode(self, passcode: str) -> bool:
        payload = self._read()
        try:
            salt = base64.b64decode(str(payload["passcode_salt"]), validate=True)
            expected = base64.b64decode(str(payload["passcode_hash"]), validate=True)
        except (KeyError, ValueError):
            return False
        actual = _passcode_hash(passcode.strip(), salt)
        return hmac.compare_digest(actual, expected)

    def save_api_key(self, api_key: str) -> None:
        normalized = api_key.strip()
        if len(normalized) < 20:
            raise LocalMeetScribeError("Gemini API key is too short.")
        if not self.passcode_configured:
            raise LocalMeetScribeError("Configure a share passcode before saving the API key.")
        payload = self._read()
        payload["version"] = 1
        payload["encrypted_api_key"] = self._protect(normalized)
        self._write(payload)

    def load_api_key(self) -> str | None:
        encrypted = self._read().get("encrypted_api_key")
        if not isinstance(encrypted, str) or not encrypted:
            return None
        try:
            return self._unprotect(encrypted).strip() or None
        except (OSError, ValueError) as exc:
            raise LocalMeetScribeError(
                "The saved Gemini API key cannot be decrypted by this Windows user."
            ) from exc

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        with contextlib.suppress(OSError):
            self.path.chmod(0o600)


def _passcode_hash(passcode: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        passcode.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )


def _protect_secret(value: str) -> str:
    raw = value.encode("utf-8")
    if os.name != "nt":
        raise LocalMeetScribeError(
            "Persistent shared API key storage requires Windows DPAPI. "
            "Use the GEMINI_API_KEY environment variable on this platform."
        )
    return f"dpapi:{base64.b64encode(_dpapi_protect(raw)).decode('ascii')}"


def _unprotect_secret(value: str) -> str:
    prefix, separator, encoded = value.partition(":")
    if not separator:
        raise ValueError("Invalid protected secret.")
    raw = base64.b64decode(encoded, validate=True)
    if prefix == "dpapi" and os.name == "nt":
        raw = _dpapi_unprotect(raw)
    else:
        raise ValueError("Unsupported protected secret.")
    return raw.decode("utf-8")


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob(value: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(value)
    return (
        _DataBlob(
            len(value),
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
        ),
        buffer,
    )


def _dpapi_protect(value: bytes) -> bytes:
    source, source_buffer = _blob(value)
    destination = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    if not crypt32.CryptProtectData(
        ctypes.byref(source),
        "LocalMeetScribe protected secret",
        None,
        None,
        None,
        0x1,
        ctypes.byref(destination),
    ):
        raise ctypes.WinError()
    del source_buffer
    return _copy_and_free(destination)


def _dpapi_unprotect(value: bytes) -> bytes:
    source, source_buffer = _blob(value)
    destination = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(destination),
    ):
        raise ctypes.WinError()
    del source_buffer
    return _copy_and_free(destination)


def _copy_and_free(blob: _DataBlob) -> bytes:
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob.pbData)
