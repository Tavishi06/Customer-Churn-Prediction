# app.py
import streamlit as st
import pandas as pd
import joblib
model = joblib.load('churn_model.joblib')

# --- Configuration ---
st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
try:
    pipeline = joblib.load("churn_pipeline.joblib")
except FileNotFoundError:
    st.error("Model pipeline file not found. Please run `train_model.py` first to create it.")
    st.stop()

# --- Helper Functions ---
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for download."""
    return df.to_csv(index=False).encode('utf-8')

# --- Main Application ---
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.header("Prediction Mode")
        prediction_mode = st.radio(
            "Choose how to predict:",
            ('Online (Single Prediction)', 'Batch (Upload CSV)')
        )
        st.markdown("---")

    # --- Page Title ---
    st.title("📞 Telco Customer Churn Predictor")
    st.markdown("This application predicts whether a customer is likely to churn based on their account details.")

    # --- Online Prediction Mode ---
    if prediction_mode == 'Online (Single Prediction)':
        st.header("Online Prediction")
        st.markdown("Enter the customer's details manually below.")
        
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            seniorcitizen = st.selectbox("Senior Citizen", ['0', '1'], help="1 for Yes, 0 for No")
            partner = st.selectbox("Has a Partner", ['Yes', 'No'])
            dependents = st.selectbox("Has Dependents", ['Yes', 'No'])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            paperlessbilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
            paymentmethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        with col2:
            phoneservice = st.selectbox("Phone Service", ['Yes', 'No'])
            multiplelines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
            internetservice = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            onlinesecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
            onlinebackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
            deviceprotection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
            techsupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
            streamingtv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
            streamingmovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        monthlycharges = st.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0, step=1.0)
        totalcharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0, step=1.0)
        
        if st.button("Predict Churn", key="predict_online"):
            user_data = {
                'gender': gender, 'SeniorCitizen': seniorcitizen, 'Partner': partner,
                'Dependents': dependents, 'tenure': tenure, 'PhoneService': phoneservice,
                'MultipleLines': multiplelines, 'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity, 'OnlineBackup': onlinebackup,
                'DeviceProtection': deviceprotection, 'TechSupport': techsupport,
                'StreamingTV': streamingtv, 'StreamingMovies': streamingmovies,
                'Contract': contract, 'PaperlessBilling': paperlessbilling,
                'PaymentMethod': paymentmethod, 'MonthlyCharges': monthlycharges,
                'TotalCharges': totalcharges
            }
            df = pd.DataFrame([user_data])
            prediction = pipeline.predict(df)[0]
            probabilities = pipeline.predict_proba(df)[0]
            
            # Probabilities array is [P(Stay), P(Churn)]
            churn_probability = probabilities[1]

            # --- THIS IS THE CORRECTED LOGIC ---
            # Now, both outcomes will display the churn probability for consistency.
            if prediction == 1:
                st.warning(f"⚠️ Prediction: Customer is likely to STAY (Churn Probability: {churn_probability*100:.2f}%)")
            else:
                st.success(f"✅ Prediction: Customer is likely to CHURN (Churn Probability: {churn_probability*100:.2f}%)")
                
            # The progress bar always shows the churn probability (risk score)
            st.progress(churn_probability)

    # --- Batch Prediction Mode ---
    # This section was already correct and did not need changes.
    if prediction_mode == 'Batch (Upload CSV)':
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with customer data to get predictions for all of them.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_df_raw = pd.read_csv(uploaded_file)
                table_placeholder = st.empty()
                table_placeholder.dataframe(batch_df_raw)

                if st.button("Predict for Batch", key="predict_batch"):
                    with st.spinner('Processing... This may take a moment.'):
                        batch_df_cleaned = batch_df_raw.copy()
                        batch_df_cleaned['TotalCharges'] = pd.to_numeric(batch_df_cleaned['TotalCharges'], errors='coerce')
                        batch_df_cleaned['TotalCharges'].fillna(0, inplace=True)
                        batch_df_cleaned['SeniorCitizen'] = batch_df_cleaned['SeniorCitizen'].astype(str)
                        
                        predictions = pipeline.predict(batch_df_cleaned)
                        probabilities = pipeline.predict_proba(batch_df_cleaned)[:, 1] # Getting churn probability
                        
                        results_df = batch_df_raw.copy()
                        results_df['Churn Prediction'] = ['Stay' if p == 1 else 'Churn' for p in predictions]
                        results_df['Churn Probability'] = [f"{p*100:.1f}%" for p in probabilities]
                        
                        st.subheader("Prediction Results")
                        table_placeholder.dataframe(results_df) 
                        csv_to_download = convert_df_to_csv(results_df)

                        st.download_button(
                           label="Download Results as CSV",
                           data=csv_to_download,
                           file_name='churn_predictions.csv',
                           mime='text/csv',
                        )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please make sure your CSV file has the correct columns and format.")

        else:
            st.info("Please upload a CSV file to proceed.")
            with st.expander("See required CSV format"):
                st.markdown("""
                Your CSV file must contain the following columns: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
                """)

if __name__ == "__main__":

    main()
