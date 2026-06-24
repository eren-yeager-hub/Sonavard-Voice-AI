import os
import librosa
import numpy as np
import tensorflow as tf
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

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
# PREDICT FUNCTION
# =============================
def predict_audio(file_path):

    audio, _ = librosa.load(file_path, sr=sr)

    # Fix length
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = librosa.util.fix_length(audio, size=target_length)

    audio = librosa.util.normalize(audio)

    features = extract_features(audio)
    features = scaler.transform(features)

    prob = model.predict(features, verbose=0)[0][0]
    label = 1 if prob > 0.5 else 0

    return label, prob

# =============================
# LOAD DATASET
# =============================
distress_folder = "segmented_dataset/emergency"
normal_folder = "segmented_dataset/normal"

y_true = []
y_pred = []
y_prob = []

def process_folder(folder, label):

    for file in os.listdir(folder):

        if not file.endswith(".wav"):
            continue

        path = os.path.join(folder, file)

        pred_label, prob = predict_audio(path)

        y_true.append(label)
        y_pred.append(pred_label)
        y_prob.append(prob)

# Run both
print("\nProcessing Distress...")
process_folder(distress_folder, 1)

print("Processing Normal...")
process_folder(normal_folder, 0)

# =============================
# METRICS
# =============================
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc = roc_auc_score(y_true, y_prob)

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

# =============================
# CLASS-WISE ACCURACY
# =============================
distress_acc = tp / (tp + fn)
normal_acc = tn / (tn + fp)

# =============================
# PRINT RESULTS
# =============================
print("\n==============================")
print("FINAL MODEL EVALUATION")
print("==============================")

print(f"\nOverall Accuracy: {acc*100:.2f}%")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC-AUC: {roc:.3f}")

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

print("\nInterpretation:")
print(f"True Negatives (Normal Correct): {tn}")
print(f"False Positives (False Alarm): {fp}")
print(f"False Negatives (Missed Distress): {fn}")
print(f"True Positives (Distress Correct): {tp}")

print("\n==============================")
print("CLASS-WISE PERFORMANCE")
print("==============================")
print(f"Distress Accuracy: {distress_acc*100:.2f}%")
print(f"Normal Accuracy: {normal_acc*100:.2f}%")

print("\n==============================")
print("DETAILED REPORT")
print("==============================")
print(classification_report(y_true, y_pred))

# =============================
# DATASET INSIGHT
# =============================
print("\n==============================")
print("DATASET ANALYSIS")
print("==============================")

total = len(y_true)
distress_count = sum(y_true)
normal_count = total - distress_count

print(f"Total Samples: {total}")
print(f"Distress Samples: {distress_count}")
print(f"Normal Samples: {normal_count}")

if distress_count < normal_count:
    print("⚠️ Dataset is imbalanced toward NORMAL")
elif distress_count > normal_count:
    print("⚠️ Dataset is imbalanced toward DISTRESS")
else:
    print("✅ Dataset is balanced")

# =============================
# FINAL VERDICT
# =============================
print("\n==============================")
print("FINAL VERDICT")
print("==============================")

if fn > fp:
    print("⚠️ Model misses distress signals (dangerous)")
elif fp > fn:
    print("⚠️ Model gives more false alarms")
else:
    print("✅ Balanced detection")

print("\nEvaluation Complete 🚀")