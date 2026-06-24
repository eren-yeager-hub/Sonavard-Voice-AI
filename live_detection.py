import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib
import soundfile as sf
import time
from collections import deque

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("distress_detector_model_v2.h5")
scaler = joblib.load("feature_scaler_v2.pkl")

# =========================
# AUDIO SETTINGS
# =========================
sr = 16000
duration = 4
target_length = sr * duration

# =========================
# THRESHOLDS (TUNED FOR DEMO)
# =========================
SILENCE_RMS = 0.02
SPEECH_RMS = 0.07
PEAK_THRESHOLD = 0.6
DISTRESS_THRESHOLD = 0.65
SUSPICIOUS_THRESHOLD = 0.35

# smoothing buffer
history = deque(maxlen=3)

# create folders
os.makedirs("collected_data/review", exist_ok=True)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_features = np.mean(mfcc.T, axis=0)
    delta_features = np.mean(delta.T, axis=0)
    delta2_features = np.mean(delta2.T, axis=0)

    return np.hstack([mfcc_features, delta_features, delta2_features]).reshape(1, -1)

# =========================
# MAIN LOOP
# =========================
print("🎤 Listening for distress... (CTRL+C to stop)")

while True:

    print("\nRecording...")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()

    audio = audio.flatten()

    # =========================
    # ENERGY CHECK
    # =========================
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))

    print(f"RMS: {rms:.4f} | Peak: {peak:.4f}")

    # Silence
    if rms < SILENCE_RMS:
        print("🔇 Silence")
        continue

    # Normal speech filter
    if rms < SPEECH_RMS:
        print("🗣️ Normal speech")
        continue

    # Not strong enough
    if peak < PEAK_THRESHOLD:
        print("⚠️ Not intense enough")
        continue

    # =========================
    # PREPROCESS
    # =========================
    audio = librosa.util.normalize(audio)

    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = librosa.util.fix_length(audio, size=target_length)

    # =========================
    # PREDICT
    # =========================
    features = extract_features(audio)
    features = scaler.transform(features)

    pred = model.predict(features, verbose=0)[0][0]

    history.append(pred)
    smooth_pred = np.mean(history)

    print(f"Raw: {pred:.3f} | Smoothed: {smooth_pred:.3f}")

    # =========================
    # SELF-LEARNING CAPTURE
    # =========================
    if (rms > 0.08 and pred < 0.40) or pred > 0.60:
        filename = f"collected_data/review/sample_{int(time.time())}.wav"
        sf.write(filename, audio, sr)
        print(f"📁 Saved for review: {filename}")

    # =========================
    # FINAL DECISION
    # =========================
    if smooth_pred > DISTRESS_THRESHOLD:
        print("🚨 DISTRESS DETECTED 🚨")
    else:
        print("✅ Normal environment")