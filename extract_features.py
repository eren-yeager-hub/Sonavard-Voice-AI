import os
import librosa
import numpy as np
import pandas as pd

dataset_path = "segmented_dataset"

features = []
labels = []

for label in ["emergency", "normal"]:

    folder = os.path.join(dataset_path, label)

    for file in os.listdir(folder):

        file_path = os.path.join(folder, file)

        try:
            audio, sr = librosa.load(file_path, sr=16000)

            # MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

            # Delta (first derivative)
            delta_mfcc = librosa.feature.delta(mfcc)

            # Delta-delta (second derivative)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            # Take mean of each feature
            mfcc_mean = np.mean(mfcc.T, axis=0)
            delta_mean = np.mean(delta_mfcc.T, axis=0)
            delta2_mean = np.mean(delta2_mfcc.T, axis=0)

            combined = np.hstack((mfcc_mean, delta_mean, delta2_mean))

            features.append(combined)
            labels.append(label)

        except Exception as e:
            print("Error processing:", file)

df = pd.DataFrame(features)
df["label"] = labels

df.to_csv("audio_features.csv", index=False)

print("Feature extraction complete")
print("Saved as audio_features.csv")