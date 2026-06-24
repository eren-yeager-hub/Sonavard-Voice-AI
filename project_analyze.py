import os
import librosa
import numpy as np

print("\n======================================")
print("AI EMERGENCY DETECTION PROJECT ANALYSIS")
print("======================================\n")

# ================================
# DATASET PATHS
# ================================

paths = [
    "dataset",
    "processed_dataset",
    "segmented_dataset",
    "collected_data"
]

# ================================
# DATASET ANALYSIS
# ================================

total_files = 0
class_counts = {}
sampling_rates = []
durations = []
formats = {}

print("Checking Datasets...\n")

for dataset_path in paths:

    if os.path.exists(dataset_path):

        print(f"Found: {dataset_path}")

        for root, dirs, files in os.walk(dataset_path):

            label = os.path.basename(root)

            for file in files:

                if file.endswith((".wav", ".mp3", ".flac")):

                    total_files += 1

                    class_counts[label] = class_counts.get(label, 0) + 1

                    ext = file.split(".")[-1]
                    formats[ext] = formats.get(ext, 0) + 1

                    try:
                        file_path = os.path.join(root, file)
                        y, sr = librosa.load(file_path, sr=None)

                        sampling_rates.append(sr)
                        durations.append(len(y) / sr)

                    except:
                        pass

# ================================
# DATASET RESULTS
# ================================

print("\n======================================")
print("DATASET SUMMARY")
print("======================================")

print("\nTotal Audio Files:", total_files)

print("\nClass Distribution:")
for k, v in class_counts.items():
    print(f"{k} : {v}")

print("\nFile Formats:")
for k, v in formats.items():
    print(f".{k} : {v}")

if sampling_rates:

    print("\nSampling Rate:")
    print("Min:", min(sampling_rates))
    print("Max:", max(sampling_rates))
    print("Most Common:", max(set(sampling_rates), key=sampling_rates.count))

if durations:

    print("\nAudio Duration:")
    print("Min:", round(min(durations),2), "sec")
    print("Max:", round(max(durations),2), "sec")
    print("Average:", round(np.mean(durations),2), "sec")


# ================================
# MODEL CHECK
# ================================

print("\n======================================")
print("MODEL FILES")
print("======================================\n")

models = []

for root, dirs, files in os.walk("."):

    for file in files:

        if file.endswith((".h5", ".pkl", ".joblib", ".pt")):
            models.append(os.path.join(root, file))

if models:

    for m in models:
        print(m)

else:
    print("No Models Found")


# ================================
# LIVE DETECTION CHECK
# ================================

print("\n======================================")
print("LIVE DETECTION FILES")
print("======================================\n")

for file in os.listdir():

    if "live_detection" in file:
        print(file)


# ================================
# TRAINING FILES
# ================================

print("\n======================================")
print("TRAINING FILES")
print("======================================\n")

training_files = [
    "train_model.py",
    "evaluate_model.py",
    "extract_features.py"
]

for file in training_files:

    if os.path.exists(file):
        print("Found:", file)


# ================================
# FEATURE FILE CHECK
# ================================

print("\n======================================")
print("FEATURE FILES")
print("======================================\n")

for file in os.listdir():

    if file.endswith(".csv"):
        print(file)


# ================================
# FINAL SUMMARY
# ================================

print("\n======================================")
print("PROJECT ANALYSIS COMPLETE")
print("======================================\n")