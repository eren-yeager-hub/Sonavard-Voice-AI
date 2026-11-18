"""
Sonavard Voice AI — Secure Real-Time Acoustic Distress & Emergency Detection Server.

Features:
- Real-time Browser WebAudio PCM streaming (/api/analyze-chunk) — works with ANY earphone/BT mic
- Optional Server-Side Native-Sample-Rate Capture (/api/server-mic/start)
- Parallel Multilingual Keyword Spotting (/api/transcript)
- Server-Sent Events (/api/stream) for zero-lag live UI updates
- Audio File Diagnostic Lab (/api/analyze-file)
- Hardened Security: CSRF protection, strict CSP & headers, rate limiting, safe defaults (127.0.0.1)
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

import config
from engine.alerts import AlertManager
from engine.audio_dsp import SAMPLE_RATE
from engine.detector import DistressDetectorEngine
from engine.security import (
    apply_security_headers,
    ensure_csrf_token,
    rate_limit,
    require_csrf,
    validate_audio_bytes,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"

# ------------------------------------------------------------------
# Event Bus (SSE Fan-Out) & In-Memory Log Ring Buffer
# ------------------------------------------------------------------
_subscribers_lock = threading.Lock()
_subscribers: list[queue.Queue] = []
_logs_lock = threading.Lock()
_logs: deque[dict] = deque(maxlen=120)

_state_lock = threading.Lock()
_monitoring_mode: str = "idle"  # "idle" | "browser_mic" | "server_mic"


def push_event(event_type: str, payload: dict) -> None:
    """Broadcast a structured JSON event to all connected SSE clients."""
    msg = {"event": event_type, "data": payload, "ts": round(time.time(), 2)}
    with _subscribers_lock:
        dead: list[queue.Queue] = []
        for q in _subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            if q in _subscribers:
                _subscribers.remove(q)


def add_log(level: str, message: str) -> None:
    entry = {
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "level": level,
        "message": str(message)[:320],
    }
    with _logs_lock:
        _logs.appendleft(entry)
    push_event("log", entry)


def _on_alert_event(event_type: str, payload: dict) -> None:
    if event_type == "log":
        add_log(payload.get("level", "info"), payload.get("message", ""))
    else:
        push_event(event_type, payload)


# ------------------------------------------------------------------
# Core Engines
# ------------------------------------------------------------------
detector = DistressDetectorEngine()
alerts = AlertManager(on_event=_on_alert_event)


@app.after_request
def _add_headers(response: Response) -> Response:
    return apply_security_headers(response)


# ------------------------------------------------------------------
# UI & Streaming Endpoints
# ------------------------------------------------------------------
@app.route("/")
def index():
    csrf_token = ensure_csrf_token()
    return render_template("index.html", csrf_token=csrf_token)


@app.route("/api/health")
def api_health():
    return jsonify(
        {
            "status": "ok",
            "model_ready": detector.is_ready,
            "model_version": detector.metadata.get("model_version", "untrained"),
        }
    )


@app.route("/api/bootstrap")
def api_bootstrap():
    csrf_token = ensure_csrf_token()
    with _state_lock:
        mode = _monitoring_mode
    with _logs_lock:
        recent_logs = list(_logs)
    return jsonify(
        {
            "csrf_token": csrf_token,
            "monitoring_mode": mode,
            "detector": detector.get_info(),
            "alerts": alerts.get_status(),
            "server_devices": detector.list_server_audio_devices(),
            "history": list(detector.score_history),
            "logs": recent_logs,
        }
    )


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events stream for real-time telemetry, alerts, and logs."""
    q: queue.Queue = queue.Queue(maxsize=64)
    with _subscribers_lock:
        _subscribers.append(q)

    def _gen():
        try:
            yield f"event: connected\ndata: {json.dumps({'ok': True})}\n\n"
            while True:
                try:
                    item = q.get(timeout=15.0)
                    yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            if _subscribers_lock is not None and _subscribers is not None:
                with _subscribers_lock:
                    if q in _subscribers:
                        _subscribers.remove(q)

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(_gen()), headers=headers)


# ------------------------------------------------------------------
# Browser Live Audio Analysis & Calibration Endpoints
# ------------------------------------------------------------------
def _decode_pcm_payload(payload: dict) -> tuple[np.ndarray, int]:
    """Decode base64 Float32 PCM (preferred) or Int16 PCM sent from WebAudio."""
    sr = int(payload.get("sample_rate", SAMPLE_RATE))
    sr = max(8000, min(96000, sr))
    fmt = str(payload.get("format", "f32")).lower()
    b64 = payload.get("pcm_b64", "")
    if not isinstance(b64, str) or not b64 or len(b64) > 2_500_000:
        raise ValueError("Invalid or oversized audio payload.")
    raw_bytes = base64.b64decode(b64, validate=True)

    if fmt == "i16":
        if len(raw_bytes) % 2 != 0:
            raise ValueError("Malformed Int16 PCM buffer.")
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        if len(raw_bytes) % 4 != 0:
            raise ValueError("Malformed Float32 PCM buffer.")
        arr = np.frombuffer(raw_bytes, dtype=np.float32).copy()

    if arr.size < 800 or arr.size > sr * 8:
        raise ValueError("Audio chunk duration out of valid bounds (0.05s - 8s).")
    return arr, sr


@app.route("/api/analyze-chunk", methods=["POST"])
@require_csrf
@rate_limit("analyze_chunk", max_requests=180, window_seconds=60.0)
def api_analyze_chunk():
    """Analyze a live 2.0s audio window captured by the browser's selected microphone."""
    if not detector.is_ready:
        return jsonify({"error": "AI models are not trained yet. Run `python train.py` first."}), 503

    payload = request.get_json(silent=True) or {}
    try:
        audio, sr = _decode_pcm_payload(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    with _state_lock:
        global _monitoring_mode
        _monitoring_mode = "browser_mic"

    result = detector.analyze_live_chunk(audio, sr=sr, run_server_stt=False)
    alert_outcome = None

    if result["distress_detected"]:
        detail = (
            f"Acoustic distress confirmed (Combined {result['smoothed_score']*100:.1f}%, "
            f"CNN {result['cnn_score']*100:.1f}%, Prosodic {result['acoustic_score']*100:.1f}%)"
        )
        add_log("alert", detail)
        alert_outcome = alerts.trigger(
            trigger_type="acoustic",
            detail=detail,
            confidence=result["smoothed_score"],
            audio_snapshot=audio if sr == SAMPLE_RATE else None,
        )

    result["alert_status"] = alerts.get_status()
    result["alert_outcome"] = alert_outcome
    push_event("telemetry", result)
    return jsonify(result)


@app.route("/api/transcript", methods=["POST"])
@require_csrf
@rate_limit("transcript", max_requests=180, window_seconds=60.0)
def api_transcript():
    """Parallel real-time multilingual keyword spotting on live speech transcripts."""
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()[:400]
    is_final = bool(payload.get("is_final", True))
    if not text:
        return jsonify({"matched": False})

    match = detector.match_keywords(text)
    alert_outcome = None
    if match:
        detail = f"Distress keyword '{match['keyword']}' heard in: \"{text[:120]}\""
        add_log("alert", detail)
        alert_outcome = alerts.trigger(
            trigger_type="keyword",
            detail=detail,
            confidence=0.98 if match["match_type"] == "exact" else 0.88,
        )
    elif is_final:
        add_log("info", f"Speech: \"{text[:140]}\"")

    resp = {
        "matched": bool(match),
        "match": match,
        "text": text,
        "is_final": is_final,
        "alert_outcome": alert_outcome,
        "alert_status": alerts.get_status(),
    }
    push_event("transcript", resp)
    return jsonify(resp)


@app.route("/api/calibrate", methods=["POST"])
@require_csrf
@rate_limit("calibrate", max_requests=30, window_seconds=60.0)
def api_calibrate():
    """Calibrate adaptive noise floor to the user's current earphone/headset mic."""
    payload = request.get_json(silent=True) or {}
    try:
        audio, sr = _decode_pcm_payload(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    res = detector.calibrate_noise_floor(audio, sr=sr)
    detector.reset_temporal_state()
    add_log("success", f"Mic calibrated — ambient noise floor: {res['noise_floor_db']} dBFS (RMS {res['noise_floor']})")
    push_event("calibrated", res)
    return jsonify(res)


@app.route("/api/session/reset", methods=["POST"])
@require_csrf
def api_session_reset():
    global _monitoring_mode
    detector.stop_server_capture()
    detector.reset_temporal_state()
    with _state_lock:
        _monitoring_mode = "idle"
    add_log("info", "Monitoring disarmed.")
    push_event("mode", {"monitoring_mode": "idle"})
    return jsonify({"status": "idle"})


# ------------------------------------------------------------------
# Optional Server-Side Mic Capture Endpoints
# ------------------------------------------------------------------
@app.route("/api/server-mic/start", methods=["POST"])
@require_csrf
@rate_limit("server_mic", max_requests=20, window_seconds=60.0)
def api_server_mic_start():
    if not detector.is_ready:
        return jsonify({"error": "AI models are not trained yet."}), 503

    payload = request.get_json(silent=True) or {}
    dev_id = payload.get("device_id")
    if dev_id is not None:
        try:
            dev_id = int(dev_id)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid device_id."}), 400

    def _on_result(res: dict, audio_16k: np.ndarray) -> None:
        if res["distress_detected"]:
            detail = f"Server Mic acoustic distress ({res['smoothed_score']*100:.1f}%)"
            add_log("alert", detail)
            alerts.trigger("acoustic", detail, confidence=res["smoothed_score"], audio_snapshot=audio_16k)
        elif res.get("keyword_match"):
            kw = res["keyword_match"]["keyword"]
            detail = f"Server Mic keyword '{kw}' detected"
            add_log("alert", detail)
            alerts.trigger("keyword", detail, confidence=0.95, audio_snapshot=audio_16k)
        res["alert_status"] = alerts.get_status()
        push_event("telemetry", res)

    def _on_error(msg: str) -> None:
        add_log("error", msg)
        with _state_lock:
            global _monitoring_mode
            _monitoring_mode = "idle"
        push_event("mode", {"monitoring_mode": "idle", "error": msg})

    ok = detector.start_server_capture(dev_id, on_result=_on_result, on_error=_on_error)
    if not ok:
        return jsonify({"error": "Could not start server microphone stream."}), 500

    with _state_lock:
        global _monitoring_mode
        _monitoring_mode = "server_mic"
    add_log("success", f"Server-side microphone stream armed (device={dev_id if dev_id is not None else 'default'}).")
    push_event("mode", {"monitoring_mode": "server_mic"})
    return jsonify({"status": "server_mic", "device_id": dev_id})


@app.route("/api/server-mic/stop", methods=["POST"])
@require_csrf
def api_server_mic_stop():
    global _monitoring_mode
    detector.stop_server_capture()
    with _state_lock:
        _monitoring_mode = "idle"
    add_log("info", "Server-side microphone stream stopped.")
    push_event("mode", {"monitoring_mode": "idle"})
    return jsonify({"status": "idle"})


# ------------------------------------------------------------------
# Audio File Diagnostic Lab Endpoint
# ------------------------------------------------------------------
@app.route("/api/analyze-file", methods=["POST"])
@require_csrf
@rate_limit("analyze_file", max_requests=25, window_seconds=60.0)
def api_analyze_file():
    if not detector.is_ready:
        return jsonify({"error": "AI models are not trained yet."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    f = request.files["file"]
    safe_name = secure_filename(f.filename or "")
    data = f.read()

    valid, msg = validate_audio_bytes(safe_name, data, config.MAX_UPLOAD_MB * 1024 * 1024)
    if not valid:
        return jsonify({"error": msg}), 400

    try:
        report = detector.analyze_audio_file(data, safe_name)
    except Exception as exc:
        return jsonify({"error": f"Failed to decode audio file: {type(exc).__name__}"}), 400

    add_log(
        "alert" if report["is_emergency"] else "info",
        f"File Lab [{safe_name}]: {report['verdict']} (Peak {report['peak_fused_score']*100:.1f}%)",
    )
    return jsonify(report)


# ------------------------------------------------------------------
# Settings & Alert Safety Controls
# ------------------------------------------------------------------
@app.route("/api/settings", methods=["POST"])
@require_csrf
@rate_limit("settings", max_requests=60, window_seconds=60.0)
def api_settings():
    payload = request.get_json(silent=True) or {}

    det_info = detector.update_settings(
        sensitivity=payload.get("sensitivity"),
        keywords=payload.get("keywords"),
        language=payload.get("language"),
        enable_server_stt=payload.get("enable_server_stt"),
    )
    alert_info = alerts.update_settings(
        dry_run=payload.get("dry_run"),
        cooldown_seconds=payload.get("cooldown_seconds"),
        cancel_window_seconds=payload.get("cancel_window_seconds"),
        save_evidence=payload.get("save_evidence"),
    )
    add_log("info", f"Settings updated (sensitivity={det_info['sensitivity']}, safe_mode={alert_info['dry_run']}).")
    return jsonify({"detector": det_info, "alerts": alert_info})


@app.route("/api/alert/sos", methods=["POST"])
@require_csrf
@rate_limit("alert_sos", max_requests=10, window_seconds=60.0)
def api_alert_sos():
    add_log("alert", "Manual SOS triggered by user!")
    outcome = alerts.trigger(
        trigger_type="manual_sos",
        detail="Manual Panic SOS activated from Sonavard Console",
        confidence=1.0,
        immediate=True,
    )
    return jsonify({"outcome": outcome, "alerts": alerts.get_status()})


@app.route("/api/alert/test", methods=["POST"])
@require_csrf
@rate_limit("alert_test", max_requests=10, window_seconds=60.0)
def api_alert_test():
    add_log("warn", "Diagnostic test alert triggered.")
    outcome = alerts.trigger(
        trigger_type="test",
        detail="Diagnostic test alert from Sonavard Voice AI",
        confidence=1.0,
        immediate=True,
    )
    return jsonify({"outcome": outcome, "alerts": alerts.get_status()})


@app.route("/api/alert/cancel", methods=["POST"])
@require_csrf
def api_alert_cancel():
    res = alerts.cancel_pending()
    if res.get("cancelled"):
        detector.reset_temporal_state()
        add_log("success", f"Pending alert {res['incident']['id']} cancelled by user.")
    return jsonify({"result": res, "alerts": alerts.get_status()})


if __name__ == "__main__":
    add_log("success", f"Sonavard Voice AI initialized (model_ready={detector.is_ready}).")
    print(f"\n[*] Sonavard Voice AI running securely on http://{config.HOST}:{config.PORT}")
    print(f"[*] Safe Mode (Dry-Run Alerts): {'ON' if alerts.dry_run else 'OFF (LIVE)'}")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True,
        use_reloader=False,
    )
