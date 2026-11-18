"""
Secure, non-blocking emergency alert dispatcher for Sonavard Voice AI.
Supports:
- Safe Mode (Dry-Run simulation so testing never spams real SMS/Emails)
- Configurable cooldown & pending countdown cancellation window
- TLS-verified SMTP email + Twilio SMS with strict timeouts
- Privacy-respecting capped evidence clips (opt-in only)
- Sanitized incident audit log (never leaks credentials)
"""

from __future__ import annotations

import smtplib
import ssl
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

import config

try:
    import geocoder
except Exception:
    geocoder = None

try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None


@dataclass
class IncidentRecord:
    id: str
    timestamp: str
    trigger_type: str  # "acoustic" | "keyword" | "manual_sos" | "test"
    detail: str
    confidence: float
    dry_run: bool
    status: str  # "pending" | "dispatched" | "cancelled" | "cooldown_suppressed"
    channels: list[str]
    evidence_file: Optional[str] = None


class AlertManager:
    """Thread-safe emergency alert coordinator with cancel window and dry-run protection."""

    def __init__(self, on_event: Optional[Callable[[str, dict], None]] = None) -> None:
        self._lock = threading.Lock()
        self._on_event = on_event

        self.dry_run: bool = config.ALERT_DRY_RUN
        self.cooldown_seconds: int = config.ALERT_COOLDOWN_SECONDS
        self.cancel_window_seconds: int = config.ALERT_CANCEL_WINDOW_SECONDS
        self.save_evidence: bool = config.SAVE_EVIDENCE_ON_EMERGENCY

        self._last_dispatch_time: float = 0.0
        self._pending_timer: Optional[threading.Timer] = None
        self._pending_incident: Optional[IncidentRecord] = None
        self._pending_audio: Optional[np.ndarray] = None
        self._pending_deadline: float = 0.0

        self._incidents: deque[IncidentRecord] = deque(maxlen=40)
        self._cached_location: Optional[tuple[float, str]] = None

    def _emit(self, event_type: str, payload: dict) -> None:
        if self._on_event:
            try:
                self._on_event(event_type, payload)
            except Exception:
                pass

    def get_status(self) -> dict:
        with self._lock:
            now = time.time()
            cooldown_remaining = max(0.0, self.cooldown_seconds - (now - self._last_dispatch_time))
            pending = None
            if self._pending_incident is not None:
                pending = {
                    **asdict(self._pending_incident),
                    "seconds_remaining": round(max(0.0, self._pending_deadline - now), 1),
                }
            return {
                "dry_run": self.dry_run,
                "cooldown_seconds": self.cooldown_seconds,
                "cooldown_remaining": round(cooldown_remaining, 1),
                "cancel_window_seconds": self.cancel_window_seconds,
                "save_evidence": self.save_evidence,
                "email_configured": config.email_is_configured(),
                "sms_configured": config.sms_is_configured(),
                "pending_alert": pending,
                "incidents": [asdict(r) for r in list(self._incidents)],
            }

    def update_settings(
        self,
        dry_run: Optional[bool] = None,
        cooldown_seconds: Optional[int] = None,
        cancel_window_seconds: Optional[int] = None,
        save_evidence: Optional[bool] = None,
    ) -> dict:
        with self._lock:
            if dry_run is not None:
                self.dry_run = bool(dry_run)
            if cooldown_seconds is not None:
                self.cooldown_seconds = max(5, min(600, int(cooldown_seconds)))
            if cancel_window_seconds is not None:
                self.cancel_window_seconds = max(0, min(30, int(cancel_window_seconds)))
            if save_evidence is not None:
                self.save_evidence = bool(save_evidence)
        return self.get_status()

    def trigger(
        self,
        trigger_type: str,
        detail: str,
        confidence: float = 1.0,
        audio_snapshot: Optional[np.ndarray] = None,
        immediate: bool = False,
    ) -> dict:
        """
        Queue or immediately dispatch an emergency alert.
        Respects cooldown (except manual SOS/test) and optional cancellation countdown.
        """
        now = time.time()
        with self._lock:
            is_manual = trigger_type in {"manual_sos", "test"}

            if not is_manual:
                if self._pending_incident is not None:
                    return {
                        "accepted": False,
                        "reason": "alert_already_pending",
                        "pending": asdict(self._pending_incident),
                    }
                if (now - self._last_dispatch_time) < self.cooldown_seconds:
                    rem = round(self.cooldown_seconds - (now - self._last_dispatch_time), 1)
                    return {"accepted": False, "reason": "cooldown_active", "cooldown_remaining": rem}

            incident_id = f"INC-{int(now * 1000) % 10000000:07d}"
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            record = IncidentRecord(
                id=incident_id,
                timestamp=ts,
                trigger_type=trigger_type,
                detail=detail[:240],
                confidence=round(float(confidence), 3),
                dry_run=self.dry_run,
                status="pending",
                channels=[],
            )

            delay = 0 if (immediate or is_manual or self.cancel_window_seconds <= 0) else self.cancel_window_seconds

            if delay == 0:
                self._last_dispatch_time = now
                record.status = "dispatched"
                self._incidents.appendleft(record)
                audio_copy = audio_snapshot.copy() if audio_snapshot is not None else None
                threading.Thread(
                    target=self._dispatch_worker,
                    args=(record, audio_copy),
                    daemon=True,
                ).start()
                self._emit("alert_dispatched", asdict(record))
                return {"accepted": True, "status": "dispatched", "incident": asdict(record)}

            # Schedule cancellable countdown
            self._pending_incident = record
            self._pending_audio = audio_snapshot.copy() if audio_snapshot is not None else None
            self._pending_deadline = now + delay
            self._pending_timer = threading.Timer(delay, self._finalize_pending, args=(incident_id,))
            self._pending_timer.daemon = True
            self._pending_timer.start()

            payload = {**asdict(record), "seconds_remaining": delay}
            self._emit("alert_pending", payload)
            return {"accepted": True, "status": "pending", "incident": payload}

    def cancel_pending(self) -> dict:
        """Abort a pending emergency alert during the countdown window."""
        with self._lock:
            if self._pending_incident is None:
                return {"cancelled": False, "reason": "no_pending_alert"}
            if self._pending_timer:
                self._pending_timer.cancel()
                self._pending_timer = None
            record = self._pending_incident
            record.status = "cancelled"
            self._incidents.appendleft(record)
            self._pending_incident = None
            self._pending_audio = None
            self._pending_deadline = 0.0
            self._emit("alert_cancelled", asdict(record))
            return {"cancelled": True, "incident": asdict(record)}

    def _finalize_pending(self, incident_id: str) -> None:
        with self._lock:
            if self._pending_incident is None or self._pending_incident.id != incident_id:
                return
            record = self._pending_incident
            audio_copy = self._pending_audio
            self._pending_incident = None
            self._pending_audio = None
            self._pending_timer = None
            self._pending_deadline = 0.0
            self._last_dispatch_time = time.time()
            record.status = "dispatched"
            self._incidents.appendleft(record)

        self._dispatch_worker(record, audio_copy)
        self._emit("alert_dispatched", asdict(record))

    def _save_evidence_clip(self, incident_id: str, audio: np.ndarray) -> Optional[str]:
        try:
            config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{incident_id}.wav"
            out_path = config.EVIDENCE_DIR / filename
            sf.write(str(out_path), np.clip(audio, -1.0, 1.0), config.SAMPLE_RATE)

            # Enforce strict retention cap for privacy & disk safety
            clips = sorted(config.EVIDENCE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
            while len(clips) > config.MAX_EVIDENCE_CLIPS:
                oldest = clips.pop(0)
                try:
                    oldest.unlink()
                except OSError:
                    pass
            return filename
        except Exception:
            return None

    def _get_location(self) -> str:
        """Best-effort cached approximate IP geolocation (5-min cache, never blocks long)."""
        now = time.time()
        if self._cached_location and (now - self._cached_location[0]) < 300:
            return self._cached_location[1]

        loc = "Location unavailable (offline or restricted)"
        if geocoder is not None:
            try:
                g = geocoder.ip("me", timeout=3.0)
                if g and g.ok and g.latlng:
                    lat, lng = g.latlng
                    loc = f"https://www.google.com/maps?q={lat:.5f},{lng:.5f}"
            except Exception:
                pass
        self._cached_location = (now, loc)
        return loc

    def _dispatch_worker(self, record: IncidentRecord, audio_snapshot: Optional[np.ndarray]) -> None:
        channels: list[str] = []

        if self.save_evidence and audio_snapshot is not None:
            saved = self._save_evidence_clip(record.id, audio_snapshot)
            if saved:
                record.evidence_file = saved
                channels.append("evidence_saved")

        if record.dry_run:
            channels.append("dry_run_simulated")
            record.channels = channels
            self._emit(
                "log",
                {
                    "level": "warn",
                    "message": f"[SAFE MODE] Simulated alert {record.id} ({record.trigger_type}: {record.detail}). No live SMS/Email sent.",
                },
            )
            return

        location = self._get_location()
        body = (
            f"SONAVARD VOICE AI — EMERGENCY ALERT\n"
            f"Incident ID: {record.id}\n"
            f"Time: {record.timestamp}\n"
            f"Trigger: {record.trigger_type.upper()}\n"
            f"Detail: {record.detail}\n"
            f"Confidence: {record.confidence * 100:.1f}%\n"
            f"Approx Location: {location}\n"
        )

        # 1. Email via TLS SMTP
        if config.email_is_configured():
            try:
                msg = MIMEText(body, "plain", "utf-8")
                msg["Subject"] = f"[EMERGENCY] Sonavard Voice AI Alert ({record.id})"
                msg["From"] = config.EMAIL_SENDER
                msg["To"] = config.EMAIL_RECEIVER

                ctx = ssl.create_default_context()
                with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as server:
                    server.ehlo()
                    server.starttls(context=ctx)
                    server.ehlo()
                    server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
                    server.sendmail(config.EMAIL_SENDER, [config.EMAIL_RECEIVER], msg.as_string())
                channels.append("email_sent")
            except Exception as exc:
                channels.append("email_failed")
                self._emit("log", {"level": "error", "message": f"Email dispatch failed: {type(exc).__name__}"})
        else:
            channels.append("email_unconfigured")

        # 2. SMS via Twilio
        if config.sms_is_configured() and TwilioClient is not None:
            try:
                client = TwilioClient(config.TWILIO_SID, config.TWILIO_TOKEN)
                client.messages.create(
                    body=body[:1500],
                    from_=config.TWILIO_PHONE,
                    to=config.TARGET_PHONE,
                )
                channels.append("sms_sent")
            except Exception as exc:
                channels.append("sms_failed")
                self._emit("log", {"level": "error", "message": f"SMS dispatch failed: {type(exc).__name__}"})
        else:
            channels.append("sms_unconfigured")

        record.channels = channels
        self._emit(
            "log",
            {"level": "alert", "message": f"Alert {record.id} finalized via: {', '.join(channels)}"},
        )
