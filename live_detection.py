import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import sounddevice as sd
import librosa
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("distress_detector_model.h5")
scaler = joblib.load("feature_scaler.pkl")

# Audio settings (same as training)
sr = 16000
duration = 3


def extract_features(audio):

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Delta
    delta = librosa.feature.delta(mfcc)

    # Delta-Delta
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Mean across time
    mfcc_features = np.mean(mfcc.T, axis=0)
    delta_features = np.mean(delta.T, axis=0)
    delta2_features = np.mean(delta2.T, axis=0)

    features = np.hstack([mfcc_features, delta_features, delta2_features])

    return features.reshape(1, -1)


print("Listening for distress sounds...")

while True:

    print("\nRecording...")

    # Record audio
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()

    audio = audio.flatten()

    # Debug amplitude
    print("Audio max amplitude:", np.max(np.abs(audio)))

    # Normalize
    audio = librosa.util.normalize(audio)
    audio = audio * 3

    # Extract features
    features = extract_features(audio)

    # Scale features
    features = scaler.transform(features)

    # Prediction
    prediction = model.predict(features, verbose=0)[0][0]

    if prediction > 0.15:
        print("🚨 DISTRESS DETECTED 🚨")
        print("Confidence:", (prediction))
    else:
        print("Normal environment")
        print("Confidence:", (prediction))