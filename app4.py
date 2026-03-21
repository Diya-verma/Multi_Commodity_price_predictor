from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- STEP 1: Define the Class exactly as it was in train.py ---
class CarDataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        X = X.copy()
        # Clean specific columns
        if 'Kms Driven' in X.columns:
            X['Kms Driven'] = pd.to_numeric(X['Kms Driven'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
        if 'Engine Displacement' in X.columns:
            X['Engine Displacement'] = pd.to_numeric(X['Engine Displacement'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
        if 'Seats' in X.columns:
            X['Seats'] = pd.to_numeric(X['Seats'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')

        # --- THE CRITICAL FIX ---
        # Only fill numeric columns with 0. Leave 'RTO' as text!
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        
        # Fill categorical columns with 'Unknown' instead of 0
        cat_cols = X.select_dtypes(exclude=['number']).columns
        X[cat_cols] = X[cat_cols].fillna('Unknown')
        
        return X

app = Flask(__name__)

# Load the model you dumped earlier
pipeline = joblib.load('car_price_pipeline.pkl')
cities = joblib.load('city_list.pkl')

def format_price(value):
    """Translates raw numbers into human-readable Lakhs/Crores."""
    if value >= 100:
        return f"₹{value/100:.2f} Crore"
    else:
        return f"₹{value:.2f} Lakh"
    
# 2. Extract Unique Cities from your Dataset
# Make sure to use the cleaned CSV you used for training
df = pd.read_csv('car_specifications.csv') 
unique_cities = sorted(df['RTO'].unique()) # Or 'RTO_City' depending on your column name

# 3. Create the Mapping (If you used LabelEncoder)
# Note: This must match the exact logic you used during training
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['RTO'])

@app.route('/')
def index():
    return render_template('index4.html', cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get raw input and clean it
            user_city = request.form.get('city_name', '').strip()
            # Check if city exists in the list used for training
            
            # Debugging: Print to your terminal to see what the user actually typed
            print(f"DEBUG: User typed -> '{user_city}'")

            # Create the DataFrame
            # Ensure 'City' matches the column name in your training df exactly!
            data = {
                'RTO': user_city if user_city else 'Unknown', 
                'Ownership': str(request.form.get('ownership')),
                'Fuel Type': str(request.form.get('fuel')),
                'Transmission': str(request.form.get('transmission')),
                'Insurance': str(request.form.get('insurance')),
                'Year of Manufacture': int(request.form.get('year', 2020)),
                'Engine Displacement': request.form.get('engine'),
                'Kms Driven': request.form.get('kms'),
                'Seats': request.form.get('seats')
            }
                            
            input_df = pd.DataFrame([data])
            
            
            print(pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out())
            # 4. Predict
            prediction = pipeline.predict(input_df)[0]
            formatted_res = format_price(prediction)
            
            
            return render_template('index4.html', result=f"PRICE: {formatted_res}", cities=cities,inputs=data)
        
        except Exception as e:
            print(f"Error: {e}") # This helps you see the REAL error in the terminal
            return render_template('index4.html', result="Error: Check your inputs!", cities=cities)
if __name__ == "__main__":
    app.run(debug=True)