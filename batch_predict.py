<<<<<<< HEAD
# batch_predict.py
# This script automates batch predictions without a user interface.

import pandas as pd
import joblib
import os

# --- Configuration ---
MODEL_PATH = 'churn_pipeline.joblib'
INPUT_CSV_PATH = 'batch_data_to_predict.csv'
OUTPUT_CSV_PATH = 'predictions_output.csv'

def run_batch_predictions():
    """
    Loads data from a predefined CSV file, makes churn predictions,
    and saves the results to a new CSV file with formatted probabilities.
    """
    print("--- Starting Automated Batch Prediction ---")

    # 1. Check for required files
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input data file not found at '{INPUT_CSV_PATH}'")
        return

    # 2. Load the trained model pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from '{MODEL_PATH}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load the batch data
    try:
        batch_df_raw = pd.read_csv(INPUT_CSV_PATH)
        print(f"✅ Input data loaded successfully from '{INPUT_CSV_PATH}'. Found {len(batch_df_raw)} rows.")
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    # 4. Preprocess (clean) the batch data
    print("⏳ Preprocessing and cleaning data...")
    batch_df_cleaned = batch_df_raw.copy()
    
    if 'TotalCharges' in batch_df_cleaned.columns:
        batch_df_cleaned['TotalCharges'] = pd.to_numeric(batch_df_cleaned['TotalCharges'], errors='coerce')
        batch_df_cleaned['TotalCharges'].fillna(0, inplace=True)
    
    if 'SeniorCitizen' in batch_df_cleaned.columns:
        batch_df_cleaned['SeniorCitizen'] = batch_df_cleaned['SeniorCitizen'].astype(str)
    
    print("✅ Data preprocessing complete.")

    # 5. Make predictions
    print("⏳ Generating predictions...")
    try:
        predictions = pipeline.predict(batch_df_cleaned)
        probabilities = pipeline.predict_proba(batch_df_cleaned)[:, 1]
        print("✅ Predictions generated successfully.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # 6. Add results to the DataFrame with formatted probability
    results_df = batch_df_raw.copy()
    results_df['Churn_Prediction'] = ['Churn' if p == 1 else 'Stay' for p in predictions]
    
    # --- THIS IS THE CORRECTED LINE ---
    results_df['Churn_Probability'] = [f"{p*100:.1f}%" for p in probabilities]

    # 7. Save the results to a new CSV file
    try:
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"🎉 --- Success! Results saved to '{OUTPUT_CSV_PATH}' ---")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

if __name__ == "__main__":
=======
# batch_predict.py
# This script automates batch predictions without a user interface.

import pandas as pd
import joblib
import os

# --- Configuration ---
MODEL_PATH = 'churn_pipeline.joblib'
INPUT_CSV_PATH = 'batch_data_to_predict.csv'
OUTPUT_CSV_PATH = 'predictions_output.csv'

def run_batch_predictions():
    """
    Loads data from a predefined CSV file, makes churn predictions,
    and saves the results to a new CSV file with formatted probabilities.
    """
    print("--- Starting Automated Batch Prediction ---")

    # 1. Check for required files
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input data file not found at '{INPUT_CSV_PATH}'")
        return

    # 2. Load the trained model pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from '{MODEL_PATH}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Load the batch data
    try:
        batch_df_raw = pd.read_csv(INPUT_CSV_PATH)
        print(f"✅ Input data loaded successfully from '{INPUT_CSV_PATH}'. Found {len(batch_df_raw)} rows.")
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    # 4. Preprocess (clean) the batch data
    print("⏳ Preprocessing and cleaning data...")
    batch_df_cleaned = batch_df_raw.copy()
    
    if 'TotalCharges' in batch_df_cleaned.columns:
        batch_df_cleaned['TotalCharges'] = pd.to_numeric(batch_df_cleaned['TotalCharges'], errors='coerce')
        batch_df_cleaned['TotalCharges'].fillna(0, inplace=True)
    
    if 'SeniorCitizen' in batch_df_cleaned.columns:
        batch_df_cleaned['SeniorCitizen'] = batch_df_cleaned['SeniorCitizen'].astype(str)
    
    print("✅ Data preprocessing complete.")

    # 5. Make predictions
    print("⏳ Generating predictions...")
    try:
        predictions = pipeline.predict(batch_df_cleaned)
        probabilities = pipeline.predict_proba(batch_df_cleaned)[:, 1]
        print("✅ Predictions generated successfully.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # 6. Add results to the DataFrame with formatted probability
    results_df = batch_df_raw.copy()
    results_df['Churn_Prediction'] = ['Churn' if p == 1 else 'Stay' for p in predictions]
    
    # --- THIS IS THE CORRECTED LINE ---
    results_df['Churn_Probability'] = [f"{p*100:.1f}%" for p in probabilities]

    # 7. Save the results to a new CSV file
    try:
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"🎉 --- Success! Results saved to '{OUTPUT_CSV_PATH}' ---")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

if __name__ == "__main__":
>>>>>>> ae9b24c5af009839ce11f23306fa70394f34b3fa
    run_batch_predictions()