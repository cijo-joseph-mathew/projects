import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('BostonHousing.csv')

# Features & Target
X = df.drop('medv', axis=1)
y = df['medv']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open('scaling.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model & Scaler saved successfully")
