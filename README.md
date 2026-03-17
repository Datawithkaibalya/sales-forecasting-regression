# Sales Forecasting using Regression

pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn

## 📊 Objective
Predict future sales using historical data.

## 🛠️ Tech Stack
Python, Pandas, Scikit-learn

## ⚙️ Workflow
- Data preprocessing
- Feature engineering
- Regression model building

## 📈 Outcome
- Improved prediction accuracy
- Supports business planning


import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data.csv")

# Basic EDA
print(df.head())
print(df.info())

# Preprocessing
df = df.dropna()

# Feature & Target
X = df.drop("target", axis=1)
y = df["target"]

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))
