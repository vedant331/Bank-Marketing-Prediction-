# Customer Churn Prediction using Machine Learning

## Overview
This project predicts whether a bank customer is likely to leave the bank (customer churn) using Machine Learning classification models.
The main objective is to analyze customer behavior and predict customer churn using classification techniques.

---

## Dataset

Dataset Used:
Bank Customer Churn Dataset (url -https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

### Features
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

### Target Variable
- `Exited`
  - `1` → Customer left the bank
  - `0` → Customer stayed

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## Project Workflow

### 1. Data Preprocessing
- Removed unnecessary columns:
  - `RowNumber`
  - `CustomerId`
  - `Surname`
- Encoded categorical columns
- Performed feature scaling using `StandardScaler`

### 2. Train-Test Split
- 70% Training Data
- 30% Testing Data

### 3. Machine Learning Models
- Random Forest Classifier

### 4. Model Evaluation
- Accuracy Score
- Classification Report

---

## Results

| Model | Accuracy |
|---|---|
| Random Forest | ~87% |

Random Forest achieved the ~87% accuracy on this dataset.

---

## Key Learnings
Through this project, I learned:
- Data preprocessing
- Feature scaling
- Classification algorithms
- Model evaluation
- End-to-end machine learning workflow

---

## Future Improvements
- Hyperparameter tuning
- Feature importance visualization
- Deploying the model using Flask or Streamlit

---

## Author
**Vedant Kadam**
