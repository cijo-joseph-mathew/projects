# import necessary libraries
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ===============================
# Load dataset
# ===============================
df = pd.read_csv('wine-quality.csv')   # <-- use your actual wine file name

print("Dataset loaded")
print(df.head())


# ===============================
# Encode target column
# ===============================
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

# Save label encoder
with open('wine_type.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Label encoder saved")


# ===============================
# Split features and target
# ===============================
X = df.drop(['type', 'type_encoded'], axis=1)
y = df['type_encoded']


# ===============================
# Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split done")


# ===============================
# Feature scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
with open('scaling.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved")


# ===============================
# Train model
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained")


# ===============================
# Save model
# ===============================
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully")
