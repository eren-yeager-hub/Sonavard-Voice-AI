# ================================
# AI Emergency Detection - FINAL PIPELINE
# ================================

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ================================
# 1. DATA LOADING + FEATURE EXTRACTION
# ================================

DATASET_PATH = "segmented_dataset"  # <-- change path

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfcc.T, axis=0)
    except:
        return None

X = []
y = []

for label, folder in enumerate(["emergency", "normal"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset Loaded:", X.shape)

# ================================
# 2. TRAIN-TEST SPLIT + SCALING
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 3. SVM MODEL
# ================================

svm = SVC(probability=True)
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_prob = svm.predict_proba(X_test)[:, 1]

# ================================
# 4. RANDOM FOREST
# ================================

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# ================================
# 5. NEURAL NETWORK
# ================================

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

nn_prob = model.predict(X_test).flatten()
nn_pred = (nn_prob > 0.5).astype(int)

# ================================
# 6. EVALUATION FUNCTION
# ================================

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{name} Results:")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Precision:", round(prec * 100, 2), "%")
    print("Recall:", round(rec * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    return cm

# ================================
# 7. RUN EVALUATION
# ================================

svm_cm = evaluate("SVM", y_test, svm_pred)
rf_cm = evaluate("Random Forest", y_test, rf_pred)
nn_cm = evaluate("Neural Network", y_test, nn_pred)

# ================================
# 8. FINAL CONFUSION MATRIX (NN)
# ================================

tn, fp, fn, tp = nn_cm.ravel()

print("\n===== FINAL MODEL (Neural Network) =====")
print("TN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)

# ================================
# 9. ROC CURVE
# ================================

plt.figure()

for name, prob in [
    ("SVM", svm_prob),
    ("RF", rf_prob),
    ("NN", nn_prob)
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}")

plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ================================
# 10. PRECISION-RECALL CURVE
# ================================

plt.figure()

for name, prob in [
    ("SVM", svm_prob),
    ("RF", rf_prob),
    ("NN", nn_prob)
]:
    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()