import librosa
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("distress_detector_model.h5")
scaler = joblib.load("feature_scaler.pkl")

sr = 16000

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc = np.mean(mfcc.T, axis=0)
    delta = np.mean(delta.T, axis=0)
    delta2 = np.mean(delta2.T, axis=0)

    features = np.hstack([mfcc, delta, delta2])
    return features.reshape(1, -1)

audio, _ = librosa.load("segmented_dataset/emergency/029586_crowd-in-panic-66480_6.wav", sr=sr)

features = extract_features(audio)
features = scaler.transform(features)

prediction = model.predict(features)[0][0]

print("Confidence:", prediction)