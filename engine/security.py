"""
Security utilities for Aegis Voice AI:
- Strict HTTP security headers (CSP, X-Frame-Options, nosniff, Permissions-Policy)
- CSRF token generation & validation + Same-Origin checks
- Sliding-window in-memory rate limiter
- Audio upload & payload sanitization
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Callable
from urllib.parse import urlparse

from flask import Request, Response, jsonify, request, session


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

# Common audio container magic prefixes (best-effort verification)
AUDIO_MAGIC_SIGNATURES = (
    b"RIFF",          # WAV
    b"ID3",           # MP3 (ID3 tag)
    b"\xff\xfb",      # MP3 frame sync
    b"\xff\xf3",      # MP3 frame sync
    b"\xff\xf2",      # MP3 frame sync
    b"OggS",          # OGG
    b"fLaC",          # FLAC
    b"\x1a\x45\xdf\xa3",  # WebM / Matroska
)


class RateLimiter:
    """Thread-safe sliding window rate limiter keyed by (ip, bucket)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def allow(self, key: str, bucket: str, max_requests: int, window_seconds: float) -> bool:
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            q = self._hits[(key, bucket)]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= max_requests:
                return False
            q.append(now)
            return True


rate_limiter = RateLimiter()


def ensure_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token or not isinstance(token, str):
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _is_same_origin(req: Request) -> bool:
    """Verify Origin or Referer matches the Flask host on mutating requests."""
    host = req.host
    origin = req.headers.get("Origin")
    if origin:
        parsed = urlparse(origin)
        return parsed.netloc == host
    referer = req.headers.get("Referer")
    if referer:
        parsed = urlparse(referer)
        return parsed.netloc == host
    # Allow direct non-browser local calls only if CSRF header matches session
    return True


def require_csrf(fn: Callable) -> Callable:
    """Decorator protecting POST/PUT/DELETE endpoints against CSRF."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _is_same_origin(request):
            return jsonify({"error": "Cross-origin request blocked."}), 403

        expected = session.get("csrf_token")
        provided = request.headers.get("X-CSRF-Token")
        if not expected or not provided or not secrets.compare_digest(expected, provided):
            return jsonify({"error": "Invalid or missing CSRF token."}), 403
        return fn(*args, **kwargs)

    return wrapper


def rate_limit(bucket: str, max_requests: int, window_seconds: float = 60.0) -> Callable:
    """Decorator enforcing per-client rate limits."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr or "local"
            if not rate_limiter.allow(client_ip, bucket, max_requests, window_seconds):
                return jsonify({"error": "Rate limit exceeded. Please wait briefly."}), 429
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def apply_security_headers(response: Response) -> Response:
    """Attach hardened HTTP security headers to every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "microphone=(self), camera=(), geolocation=(), payment=(), usb=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "media-src 'self' blob:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


def validate_audio_bytes(filename: str, data: bytes, max_bytes: int) -> tuple[bool, str]:
    """Validate uploaded audio filename, size, and magic header."""
    if not filename:
        return False, "Missing filename."

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return False, f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}."

    if not data:
        return False, "Uploaded file is empty."

    if len(data) > max_bytes:
        return False, f"File exceeds maximum size of {max_bytes // (1024 * 1024)} MB."

    header = data[:16]
    is_mp4_m4a = len(header) >= 8 and header[4:8] == b"ftyp"
    if not (is_mp4_m4a or any(header.startswith(sig) for sig in AUDIO_MAGIC_SIGNATURES)):
        return False, "File signature does not match a valid audio container."

    return True, "ok"
