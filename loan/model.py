import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Loan.csv")

# Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Drop Loan_ID
df.drop('Loan_ID', axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
cat_cols = [
    'Gender', 'Married', 'Dependents',
    'Education', 'Self_Employed',
    'Property_Area', 'Loan_Status'
]

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Split features & target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model & scaler saved successfully")
