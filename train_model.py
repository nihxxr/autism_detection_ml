
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load local dataset
df = pd.read_csv('model/autism_data.csv')

# Data Cleaning
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Encoding categorical variables
df.replace({'yes': 1, 'no': 0, 'm': 1, 'f': 0, 'Male': 1, 'Female': 0}, inplace=True)

# Features and target
X = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'jaundice', 'family_mem_with_ASD']]
y = df['Class/ASD']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/model.pkl')

# Test accuracy
predictions = model.predict(X_test)
print("Model trained. Accuracy:", accuracy_score(y_test, predictions))
