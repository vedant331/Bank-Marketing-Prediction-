# ==============================
# Customer Churn Prediction
# ==============================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==============================
# Load Dataset
# ==============================

path = "Churn_Modelling.csv"

df = pd.read_csv(path)

dataset = pd.DataFrame(df)

print(dataset.head())

# ==============================
# Data Preprocessing
# ==============================

# Fill missing values
dataset.fillna(0, inplace=True)

# Encode categorical columns
le = LabelEncoder()

dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset['Geography'] = le.fit_transform(dataset['Geography'])

# ==============================
# Feature Selection
# ==============================

X = dataset.drop(
    ['RowNumber', 'CustomerId', 'Surname', 'Exited'],
    axis=1
)

y = dataset['Exited']

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# Random Forest Model
# ==============================

radF = RandomForestClassifier()

radF.fit(X_train, y_train)

# ==============================
# Prediction
# ==============================

y_pred = radF.predict(X_test)

# ==============================
# Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==============================
# Feature Importance
# ==============================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": radF.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop Feature Importance:\n")
print(importance.head(10))
