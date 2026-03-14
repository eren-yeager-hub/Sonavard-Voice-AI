import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# load feature dataset
data = pd.read_csv("audio_features.csv")

# remove label column
X = data.drop("label", axis=1)

# fit scaler
scaler = StandardScaler()
scaler.fit(X)

# save scaler
joblib.dump(scaler, "feature_scaler.pkl")

print("Scaler saved as feature_scaler.pkl")