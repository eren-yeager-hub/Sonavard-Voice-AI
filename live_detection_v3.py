import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import librosa
import tensorflow as tf
import joblib
import soundfile as sf
import time
import speech_recognition as sr
from collections import deque
from alert_system import send_alert


# =========================
# LOG SYSTEM (FOR FRONTEND)
# =========================
logs = []

def add_log(text):
    print(text)
    logs.append(text)


# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("distress_detector_model_v2.h5")
scaler = joblib.load("feature_scaler_v2.pkl")


# =========================
# AUDIO SETTINGS
# =========================
SAMPLE_RATE = 16000
DURATION = 4
TARGET_LENGTH = SAMPLE_RATE * DURATION


# =========================
# THRESHOLDS
# =========================
SILENCE_RMS = 0.02
SPEECH_RMS = 0.07
PEAK_THRESHOLD = 0.6
DISTRESS_THRESHOLD = 0.65


# smoothing buffer
history = deque(maxlen=4)


# cooldown
last_alert_time = 0
ALERT_COOLDOWN = 10


# create folders
os.makedirs("collected_data/review", exist_ok=True)


# =========================
# KEYWORDS
# =========================
keywords = [
    "help",
    "help me",
    "emergency",
    "call ambulance",
    "bachao",
    "bachao bachao",
    "save me",
    "madat karo",
    "ambulance"
]


# =========================
# SAVE AUDIO
# =========================
def save_audio(audio, label="sample"):

    filename = f"collected_data/review/{label}_{int(time.time())}.wav"
    sf.write(filename, audio, SAMPLE_RATE)

    add_log(f"📁 Saved: {filename}")


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(audio):

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_features = np.mean(mfcc.T, axis=0)
    delta_features = np.mean(delta.T, axis=0)
    delta2_features = np.mean(delta2.T, axis=0)

    return np.hstack([
        mfcc_features,
        delta_features,
        delta2_features
    ]).reshape(1, -1)


# =========================
# DISTRESS DETECTION
# =========================
def detect_distress(audio):

    global history

    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))

    add_log(f"RMS: {rms:.4f} | Peak: {peak:.4f}")

    if rms < SILENCE_RMS:
        add_log("🔇 Silence")
        return False

    if rms < SPEECH_RMS:
        add_log("🗣️ Normal speech")
        return False

    if peak < PEAK_THRESHOLD:
        add_log("⚠️ Not intense enough")
        return False

    audio = librosa.util.normalize(audio)

    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]
    else:
        audio = librosa.util.fix_length(audio, size=TARGET_LENGTH)

    features = extract_features(audio)
    features = scaler.transform(features)

    pred = model.predict(features, verbose=0)[0][0]

    history.append(pred)
    smooth_pred = np.mean(history)

    add_log(f"Raw: {pred:.3f} | Smoothed: {smooth_pred:.3f}")

    if smooth_pred > DISTRESS_THRESHOLD:
        return True

    return False


# =========================
# KEYWORD DETECTION (IMPROVED)
# =========================
def detect_keywords(audio):

    recognizer = sr.Recognizer()

    try:

        audio_int16 = (audio * 32767).astype(np.int16)

        audio_data = sr.AudioData(
            audio_int16.tobytes(),
            SAMPLE_RATE,
            2
        )

        # Try multiple times for better accuracy
        for _ in range(2):

            text = recognizer.recognize_google(audio_data)

            add_log(f"🗣️ Speech: {text}")

            for word in keywords:

                if word in text.lower():

                    add_log("🚨 Keyword Detected")
                    return True

    except sr.UnknownValueError:
        add_log("❓ Could not understand speech")

    except sr.RequestError:
        add_log("⚠️ API Error")

    return False


# =========================
# EMERGENCY
# =========================
def trigger_emergency():

    global last_alert_time

    if time.time() - last_alert_time < ALERT_COOLDOWN:

        add_log("⏳ Cooldown active...")
        return False

    last_alert_time = time.time()

    add_log("\n🚨🚨🚨 EMERGENCY DETECTED 🚨🚨🚨\n")

    send_alert()

    add_log("📧 Email Sent")
    add_log("📱 SMS Sent")

    return True


# =========================
# MAIN SERVER FUNCTION
# =========================
def run_detection(audio, phase="distress"):

    audio = np.array(audio).flatten()

    if phase == "distress":

        add_log("\n🎯 Distress Detection Phase")

        save_audio(audio, "distress")

        distress_detected = detect_distress(audio)

        if distress_detected:

            add_log("🚨 Distress Detected")

            alert = trigger_emergency()

            return {
                "status": "emergency",
                "phase": "distress",
                "message": "🚨 Distress Detected - Alert Sent" if alert else "⏳ Cooldown Active",
                "logs": logs
            }

        else:

            add_log("✅ No distress")

            return {
                "status": "ok",
                "phase": "distress",
                "message": "No Distress",
                "logs": logs
            }


    elif phase == "keyword":

        add_log("\n🎯 Keyword Detection Phase")

        save_audio(audio, "keyword")

        keyword_detected = detect_keywords(audio)

        if keyword_detected:

            alert = trigger_emergency()

            return {
                "status": "emergency",
                "phase": "keyword",
                "message": "🚨 Keyword Detected - Alert Sent" if alert else "⏳ Cooldown Active",
                "logs": logs
            }

        else:

            add_log("✅ No keyword detected")

            return {
                "status": "ok",
                "phase": "keyword",
                "message": "No Keyword",
                "logs": logs
            }


# =========================
# DETECT FUNCTION (UPDATED)
# =========================
def detect(phase):

    import sounddevice as sd

    global logs
    logs = []

    if phase == "distress":
        add_log("\n🎤 Listening (Distress Phase - 4s)...")

    else:
        add_log("\n🎤 Listening (Keyword Phase - 4s)...")

    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1
    )

    sd.wait()

    result = run_detection(audio, phase=phase)

    return result
