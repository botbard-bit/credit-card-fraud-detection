from flask import Flask, request, jsonify, render_template 
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load models and preprocessors
try:
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
except Exception as e:
    print(f"Error loading models: {e}. Please run credit_risk_model.py first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    data = request.json
    
    try:
        # 1. Map CIBIL Score to Credit_History
        # Rule of thumb: > 750 = Excellent, 650-750 = Good, < 650 = Poor
        cibil_score = int(data.get('cibil_score', 700))
        if cibil_score > 750:
            credit_history = 'Excellent'
        elif cibil_score >= 650:
            credit_history = 'Good'
        else:
            credit_history = 'Poor'

        # 2. Prepare the input data
        # Using defaults for features not explicitly in the primary form
        user_data = {
            'Age': int(data.get('age', 30)),
            'Income': int(data.get('income', 50000)),
            'Credit_Amount': int(data.get('credit_amount', 15000)),
            'Loan_Duration': int(data.get('loan_duration', 24)),
            'Employment_Status': data.get('employment_status', '1-4 yrs'),
            'Credit_History': credit_history,
            'Existing_Loans': int(data.get('existing_loans', 1))
        }
        
        df_new = pd.DataFrame([user_data])
        
        # Encode categorical features
        for col in ['Employment_Status', 'Credit_History']:
            try:
                df_new[col] = encoders[col].transform(df_new[col])
            except ValueError:
                df_new[col] = 0 # Fallback
                
        # Reorder columns to match training set
        df_new = df_new[feature_cols]
        
        # Scale features
        df_scaled = scaler.transform(df_new)
        
        # Predict
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]
        
        result = "Default (High Risk)" if pred == 1 else "No Default (Low Risk)"
        
        return jsonify({
            'prediction': result,
            'probability': round(prob * 100, 2),
            'risk_level': 'high' if pred == 1 else 'low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
