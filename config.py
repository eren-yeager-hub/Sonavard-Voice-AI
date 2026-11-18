"""
Secure configuration management for Sonavard Voice AI.
Loads environment variables from a local .env file (if present) without
requiring external dotenv packages and never exposes secrets to the client.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(env_path: Path) -> None:
    """Minimal, safe .env file loader (does not overwrite existing env vars)."""
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0, maximum: int = 100000) -> int:
    try:
        val = int(os.getenv(name, str(default)))
        return max(minimum, min(maximum, val))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1e6) -> float:
    try:
        val = float(os.getenv(name, str(default)))
        return max(minimum, min(maximum, val))
    except ValueError:
        return default


# --- Server & Security Settings ---
HOST: str = os.getenv("SONAVARD_HOST", "127.0.0.1")
PORT: int = _env_int("SONAVARD_PORT", 5000, minimum=1, maximum=65535)
DEBUG: bool = _env_bool("SONAVARD_DEBUG", False)
SECRET_KEY: str = os.getenv("SONAVARD_SECRET_KEY") or secrets.token_hex(32)
MAX_UPLOAD_MB: int = _env_int("SONAVARD_MAX_UPLOAD_MB", 10, minimum=1, maximum=50)

# --- Model & Paths ---
MODELS_DIR: Path = BASE_DIR / "models"
CNN_MODEL_PATH: Path = MODELS_DIR / "distress_cnn.keras"
ENSEMBLE_MODEL_PATH: Path = MODELS_DIR / "acoustic_ensemble.joblib"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

EVIDENCE_DIR: Path = BASE_DIR / "evidence_clips"
MAX_EVIDENCE_CLIPS: int = _env_int("SONAVARD_MAX_EVIDENCE_CLIPS", 20, minimum=1, maximum=200)
SAVE_EVIDENCE_ON_EMERGENCY: bool = _env_bool("SONAVARD_SAVE_EVIDENCE", False)

# --- Audio DSP Constants ---
SAMPLE_RATE: int = 16000
WINDOW_SECONDS: float = 2.0
TARGET_SAMPLES: int = int(SAMPLE_RATE * WINDOW_SECONDS)

# --- Alert & Safety Settings ---
ALERT_COOLDOWN_SECONDS: int = _env_int("SONAVARD_ALERT_COOLDOWN", 25, minimum=5, maximum=600)
ALERT_DRY_RUN: bool = _env_bool("SONAVARD_ALERT_DRY_RUN", True)
ALERT_CANCEL_WINDOW_SECONDS: int = _env_int("SONAVARD_CANCEL_WINDOW", 5, minimum=0, maximum=30)

# Email (SMTP TLS)
SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = _env_int("SMTP_PORT", 587, minimum=1, maximum=65535)
EMAIL_SENDER: str = os.getenv("EMAIL_SENDER", "").strip()
EMAIL_RECEIVER: str = os.getenv("EMAIL_RECEIVER", "").strip()
EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "").strip()

# Twilio SMS
TWILIO_SID: str = os.getenv("TWILIO_SID", "").strip()
TWILIO_TOKEN: str = os.getenv("TWILIO_TOKEN", "").strip()
TWILIO_PHONE: str = os.getenv("TWILIO_PHONE", "").strip()
TARGET_PHONE: str = os.getenv("TARGET_PHONE", "").strip()

# Default Multilingual Distress Keywords
DEFAULT_KEYWORDS: list[str] = [
    # English
    "help",
    "help me",
    "save me",
    "emergency",
    "call ambulance",
    "ambulance",
    "call police",
    "police",
    "stop it",
    "let me go",
    "somebody help",
    "fire",
    "attack",
    "sos",
    # Hindi / Hinglish
    "bachao",
    "bacha lo",
    "madad",
    "madat",
    "madad karo",
    "madat karo",
    "chhodo",
    "chhod do",
    "khatra",
    "police bulao",
    # Spanish
    "ayuda",
    "socorro",
    "auxilio",
    "emergencia",
    # French
    "au secours",
    "aidez-moi",
]


def email_is_configured() -> bool:
    return bool(EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER)


def sms_is_configured() -> bool:
    return bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_PHONE and TARGET_PHONE)
