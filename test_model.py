import os
import librosa
import numpy as np
import tensorflow as tf
import joblib

# =============================
# LOAD MODEL
# =============================
model = tf.keras.models.load_model("distress_detector_model_v2.h5")
scaler = joblib.load("feature_scaler_v2.pkl")

# =============================
# SETTINGS
# =============================
sr = 16000
duration = 4
target_length = sr * duration

# =============================
# FEATURE EXTRACTION
# =============================
def extract_features(audio):

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_features = np.mean(mfcc.T, axis=0)
    delta_features = np.mean(delta.T, axis=0)
    delta2_features = np.mean(delta2.T, axis=0)

    features = np.hstack([
        mfcc_features,
        delta_features,
        delta2_features
    ])

    return features.reshape(1, -1)


# =============================
# TEST SINGLE AUDIO
# =============================
def test_audio(file_path):

    print("\nTesting:", file_path)

    audio, _ = librosa.load(file_path, sr=sr)

    # Fix length
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = librosa.util.fix_length(audio, size=target_length)

    audio = librosa.util.normalize(audio)

    features = extract_features(audio)
    features = scaler.transform(features)

    prediction = model.predict(features, verbose=0)[0][0]

    print("Confidence:", round(float(prediction), 3))

    if prediction > 0.5:
        print("🚨 DISTRESS")
        return 1, prediction
    else:
        print("✅ NORMAL")
        return 0, prediction


# =============================
# TEST MULTIPLE FILES
# =============================
def test_list(file_list, label_name):

    print("\n==============================")
    print(f"Testing {label_name}")
    print("==============================")

    scores = []

    for file in file_list:

        _, pred = test_audio(file)
        scores.append(pred)

    print("\nAverage Confidence:", round(float(np.mean(scores)), 3))


# =============================
# YOUR TEST FILES
# =============================

# 🔴 Dataset Distress
dataset_distress = [
    "segmented_dataset/emergency/10.wav",
    "segmented_dataset/emergency/029586_crowd-in-panic-66480_16.wav",
]

# 🟢 Dataset Normal
dataset_normal = [
    "segmented_dataset/normal/talking-people-6368_10.wav",
]

# 🔥 Your Real Voice (PUT YOUR FILES HERE)
real_test = [
    "collected_data/review/distress_1774683218.wav",
    "collected_data/review/sample_1774099426.wav",
    "collected_data/review/distress_1774771160.wav"
]

# =============================
# RUN TESTS
# =============================

test_list(dataset_distress, "Dataset Distress")
test_list(dataset_normal, "Dataset Normal")
test_list(real_test, "Your Real Voice")