import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib
import soundfile as sf
import time
import speech_recognition as sr
from collections import deque
from alert_system import send_alert

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
history = deque(maxlen=3)

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

    print(f"📁 Saved: {filename}")


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
# RECORD AUDIO
# =========================
def record_audio():

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    sd.wait()
    return audio.flatten()


# =========================
# DISTRESS DETECTION
# =========================
def detect_distress(audio):

    global history

    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))

    print(f"RMS: {rms:.4f} | Peak: {peak:.4f}")

    if rms < SILENCE_RMS:
        print("🔇 Silence")
        return False

    if rms < SPEECH_RMS:
        print("🗣️ Normal speech")
        return False

    if peak < PEAK_THRESHOLD:
        print("⚠️ Not intense enough")
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

    print(f"Raw: {pred:.3f} | Smoothed: {smooth_pred:.3f}")

    if smooth_pred > DISTRESS_THRESHOLD:
        return True

    return False


# =========================
# KEYWORD DETECTION
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

        text = recognizer.recognize_google(audio_data)

        print(f"🗣️ Speech: {text}")

        for word in keywords:

            if word in text.lower():

                print("🚨 Keyword Detected")
                return True

    except sr.UnknownValueError:
        pass

    except sr.RequestError:
        print("⚠️ API Error")

    return False


# =========================
# EMERGENCY
# =========================
last_alert_time = 0

def trigger_emergency():

    global last_alert_time

    if time.time() - last_alert_time < ALERT_COOLDOWN:

        print("⏳ Cooldown active...")
        return

    last_alert_time = time.time()

    print("\n🚨🚨🚨 EMERGENCY DETECTED 🚨🚨🚨\n")

    send_alert()  

# =========================
# MAIN LOOP (only run if script is executed directly)
# =========================
if __name__ == "_main_":

    print("\n🎤 AI Emergency Detection Started")
    print("Distress (4s) → Keywords (4s)")
    print("CTRL+C to stop\n")

    while True:

        # =========================
        # DISTRESS PHASE
        # =========================
        print("\n🎯 Distress Detection Phase (4s)")

        distress_audio = record_audio()

        save_audio(distress_audio, "distress")

        distress_detected = detect_distress(distress_audio)

        if distress_detected:

            print("🚨 Distress Detected")
            trigger_emergency()
            continue

        else:
            print("✅ No distress")


        # =========================
        # KEYWORD PHASE
        # =========================
        print("\n🎯 Keyword Detection Phase (4s)")

        keyword_audio = record_audio()

        save_audio(keyword_audio, "keyword")

        keyword_detected = detect_keywords(keyword_audio)

        if keyword_detected:

            trigger_emergency()

        else:

            print("✅ No keyword detected")


        
        