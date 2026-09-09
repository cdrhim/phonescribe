from __future__ import annotations

import base64
import contextlib
import ctypes
import hashlib
import hmac
import json
import math
import os
import secrets
import tempfile
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from local_meetscribe.utils.errors import LocalMeetScribeError

PBKDF2_ITERATIONS = 240_000
REMOTE_SESSION_FILE_VERSION = 1
_REMOTE_SESSION_FILE_LOCK = threading.Lock()


class RemoteSessionStore:
    """Durable bearer sessions stored as one-way token hashes.

    The caller receives the random bearer token exactly once. Only its SHA-256
    digest and absolute expiry are persisted, so copying the data directory
    cannot reveal an active bearer credential.
    """

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "config" / "remote_sessions.json"
        self._lock = _REMOTE_SESSION_FILE_LOCK

    def issue(self, ttl_sec: int) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        token_hash = _remote_session_hash(token)
        with self._lock:
            sessions = self._read_sessions()
            _prune_remote_sessions(sessions, now)
            sessions[token_hash] = now + max(1, ttl_sec)
            self._write_sessions(sessions)
        return token

    def is_valid(self, token: str) -> bool:
        if not token:
            return False
        now = time.time()
        token_hash = _remote_session_hash(token)
        with self._lock:
            sessions = self._read_sessions()
            changed = _prune_remote_sessions(sessions, now)
            expires_at = sessions.get(token_hash)
            if changed:
                self._write_sessions(sessions)
        return expires_at is not None and expires_at > now

    def clear(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)

    def _read_sessions(self) -> dict[str, float]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("version") != REMOTE_SESSION_FILE_VERSION:
            return {}
        raw_sessions = payload.get("sessions")
        if not isinstance(raw_sessions, dict):
            return {}
        sessions: dict[str, float] = {}
        for token_hash, raw_expiry in raw_sessions.items():
            if not isinstance(token_hash, str) or not _is_sha256_hex(token_hash):
                continue
            if isinstance(raw_expiry, bool) or not isinstance(raw_expiry, (int, float)):
                continue
            expires_at = float(raw_expiry)
            if math.isfinite(expires_at):
                sessions[token_hash] = expires_at
        return sessions

    def _write_sessions(self, sessions: dict[str, float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": REMOTE_SESSION_FILE_VERSION,
            "sessions": sessions,
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            with contextlib.suppress(OSError):
                temporary_path.chmod(0o600)
            os.replace(temporary_path, self.path)
            temporary_path = None
            with contextlib.suppress(OSError):
                self.path.chmod(0o600)
        finally:
            if temporary_path is not None:
                with contextlib.suppress(OSError):
                    temporary_path.unlink()


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
        # Invalidate old bearer sessions before committing the new passcode so a
        # failed session-file update cannot leave credentials issued under the
        # previous passcode active.
        RemoteSessionStore(self.path.parent.parent).clear()
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


def _remote_session_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _prune_remote_sessions(sessions: dict[str, float], now: float) -> bool:
    expired = [token_hash for token_hash, expires_at in sessions.items() if expires_at <= now]
    for token_hash in expired:
        sessions.pop(token_hash, None)
    return bool(expired)


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
