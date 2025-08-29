<<<<<<< HEAD
# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("--- Starting Model Training ---")

# 1. Load Data
# Assuming 'churn.csv' is in the same directory
try:
    df = pd.read_csv("./churn.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'churn.csv' not found. Please place it in the same directory.")
    exit()

# 2. Initial Data Cleaning
# Drop customerID, as it's not a predictive feature
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric, handling errors and filling missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Ensure binary features are strings for consistent processing
df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

# Define target and features
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Define Preprocessing Steps for Numerical and Categorical Features
# Identify column types
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Define the Model
# Using the best parameters found from your notebook's hyperparameter search
model = LogisticRegression(solver='liblinear', C=0.1)

# 5. Create the Full Prediction Pipeline
# This pipeline bundles preprocessing and the model into one step
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# 6. Split Data and Train the Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the full pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# 7. Evaluate the Pipeline
print("\n--- Model Evaluation ---")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# 8. Save the Final Pipeline
# This single file contains everything: preprocessing and the trained model
filename = 'churn_pipeline.joblib'
joblib.dump(pipeline, filename)
=======
# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("--- Starting Model Training ---")

# 1. Load Data
# Assuming 'churn.csv' is in the same directory
try:
    df = pd.read_csv("./churn.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'churn.csv' not found. Please place it in the same directory.")
    exit()

# 2. Initial Data Cleaning
# Drop customerID, as it's not a predictive feature
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric, handling errors and filling missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Ensure binary features are strings for consistent processing
df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

# Define target and features
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Define Preprocessing Steps for Numerical and Categorical Features
# Identify column types
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Create preprocessing pipelines for both numeric and categorical data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Define the Model
# Using the best parameters found from your notebook's hyperparameter search
model = LogisticRegression(solver='liblinear', C=0.1)

# 5. Create the Full Prediction Pipeline
# This pipeline bundles preprocessing and the model into one step
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# 6. Split Data and Train the Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the full pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete.")

# 7. Evaluate the Pipeline
print("\n--- Model Evaluation ---")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# 8. Save the Final Pipeline
# This single file contains everything: preprocessing and the trained model
filename = 'churn_pipeline.joblib'
joblib.dump(pipeline, filename)
>>>>>>> ae9b24c5af009839ce11f23306fa70394f34b3fa
print(f"\nModel pipeline saved successfully as '{filename}'")