import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("data.csv")

# Drop non-informative columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handle missing values
imputer = SimpleImputer(strategy="mean")
df["Age"] = imputer.fit_transform(df[["Age"]])
df["Fare"] = imputer.fit_transform(df[["Fare"]])
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# Encode categorical variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# Split into features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save the model
joblib.dump(model, "model/titanic_model.pkl")
print("Model saved to model/titanic_model.pkl")
