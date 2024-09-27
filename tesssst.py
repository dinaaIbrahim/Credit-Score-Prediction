import pandas as pd
import numpy as np
import category_encoders as ce
import pickle
from flask import Flask, request, jsonify,render_template


app = Flask(__name__)

# Load the XGBoost model
def load_xgb_model(model_file_path):
    with open(model_file_path, 'rb') as model_file:
        xgb_model = pickle.load(model_file)
    return xgb_model

# Load the fitted TargetEncoder
def load_encoder(encoder_file_path):
    with open(encoder_file_path, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    return encoder

def load_scaler(scaler_file_path):
    with open(scaler_file_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler
# Define the file paths
xgb_model_path = r"C:\Users\dinai\Downloads\Credit_Score_Prediction\xgboost_model.pkl"
encoder_path = r"C:\Users\dinai\Downloads\Credit_Score_Prediction\target_encoder.pkl"
scaler_path = r"C:\Users\dinai\Downloads\Credit_Score_Prediction\scaler.pkl"

# Load the XGBoost model, encoder, and scaler from the files
loaded_xgb_model = load_xgb_model(xgb_model_path)
loaded_encoder = load_encoder(encoder_path)
loaded_scaler = load_scaler(scaler_path)


# Define the mapping from numerical predictions to categorical labels
prediction_mapping = {0: 'Standard', 1: 'Poor', 2: 'Good'}

def preprocess_classify_data(data, encoder, scaler, model):
    # List of columns to convert
    columns_to_convert = [
        'Annual_Income',
        'Num_of_Loan',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit',
        'Outstanding_Debt',
        'Amount_invested_monthly',
        'Monthly_Balance',
        'Age'
    ]

    # Convert specified columns from object to float, coercing errors to NaN
    data[columns_to_convert] = data[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Drop specified columns
    columns_to_delete = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Occupation', 'Credit_History_Age']
    data.drop(columns=columns_to_delete, inplace=True)

    # Specify categorical columns
    original_categorical_cols = [
        'Type_of_Loan',
        'Credit_Mix',
        'Payment_of_Min_Amount',
        'Payment_Behaviour'
    ]

    # Transform the data using the loaded encoder
    data_encoded = encoder.transform(data[original_categorical_cols])

    # Drop the original categorical columns
    data.drop(original_categorical_cols, axis=1, inplace=True)

    # Concatenate the original data with the encoded features
    data_encoded = pd.concat([data, data_encoded], axis=1)

    # Fill null values with the mean for additional numerical columns
    additional_numeric_cols = ['Changed_Credit_Limit', 'Outstanding_Debt']
    for col in additional_numeric_cols:
        data_encoded[col].fillna(data_encoded[col].mean(), inplace=True)

    # Scale the data using the loaded scaler
    data_scaled = scaler.transform(data_encoded)

    # Apply the model on the preprocessed and scaled data
    predictions = model.predict(data_scaled)

    # Map numerical predictions to categorical labels
    mapped_predictions = [prediction_mapping[pred] for pred in predictions]

    return mapped_predictions


@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        input_data = pd.DataFrame([form_data])

        predictions = preprocess_classify_data(input_data, loaded_encoder, loaded_scaler, loaded_xgb_model)
        return render_template('predict.html', prediction=predictions[0])

if __name__ == '__main__':
    app.run(debug=True)















