import os
import joblib
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.base import BaseEstimator, TransformerMixin




# Add this helper function at the top of main.py
def safe_float(value, default=0.0):
    try:
        return float(value) if value and value.strip() else default
    except (ValueError, TypeError):
        return default

# --- 1. CUSTOM CLASS FOR CAR PIPELINE (CRITICAL) ---
class CarDataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'Kms Driven' in X.columns:
            X['Kms Driven'] = pd.to_numeric(X['Kms Driven'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
        if 'Engine Displacement' in X.columns:
            X['Engine Displacement'] = pd.to_numeric(X['Engine Displacement'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
        if 'Seats' in X.columns:
            X['Seats'] = pd.to_numeric(X['Seats'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        cat_cols = X.select_dtypes(exclude=['number']).columns
        X[cat_cols] = X[cat_cols].fillna('Unknown')
        return X

app = Flask(__name__)
base_path = os.path.dirname(os.path.abspath(__file__))

# --- 2. LOAD ALL MODELS AND ENCODERS ---
models = {
    'laptop': joblib.load(os.path.join(base_path, 'laptop_model.pkl')),
    'mobile': joblib.load(os.path.join(base_path, 'mobile_price_model.pkl')),
    'car': joblib.load(os.path.join(base_path, 'car_price_pipeline.pkl')),
    'bike': joblib.load(os.path.join(base_path, 'bike_model.pkl'))
}

# Encoders & Data Lists
le_brand = joblib.load(os.path.join(base_path, 'brand_encoder.pkl'))
le_cpu_brand = joblib.load(os.path.join(base_path, 'cpu_brand_encoder.pkl'))
le_cpu_tier = joblib.load(os.path.join(base_path, 'cpu_tier_encoder.pkl'))
cities = joblib.load(os.path.join(base_path, 'city_list.pkl'))
bike_df = pd.read_csv(os.path.join(base_path, 'bike_specifications.csv'))

# --- 3. HELPERS ---
def format_car_price(value):
    if value >= 100: return f"₹{value/100:.2f} Crore"
    return f"₹{value:.2f} Lakh"

def get_dropdown_data():
    return {
        'brands': le_brand.classes_, 
        'cpu_brands': le_cpu_brand.classes_,
        'cpu_tiers': le_cpu_tier.classes_,
        'cities': cities,
        'bike_names': sorted(bike_df['bike_name'].unique()),
        'tyre_types': sorted(bike_df['Tyre_Type'].fillna('Unknown').unique().astype(str))
    }
# --- ROUTES TO SHOW THE FORMS ---

@app.route('/')
def home():
    # This opens your main dashboard/home page
    return render_template('home.html')

@app.route('/laptop')
def laptop_page():
    return render_template('index.html', **get_dropdown_data())

@app.route('/mobile')
def mobile_page():
    return render_template('index2.html')

@app.route('/bike')
def bike_page():
    return render_template('index3.html', **get_dropdown_data())

@app.route('/car')
def car_page():
    return render_template('index4.html', **get_dropdown_data())

# --- THE PREDICTION ROUTE (Keep your existing one) ---
@app.route('/predict/<category>', methods=['POST'])
def predict(category):
    # Mapping categories to their specific HTML files
    template_map = {
        'laptop': 'index.html',
        'mobile': 'index2.html',
        'bike': 'index3.html',
        'car': 'index4.html'
    }
    
    target_file = template_map.get(category, 'home.html')
    dropdown_data = get_dropdown_data()
    # ... (Keep all the logic we wrote in the previous step) ...
    final_price_string = ""
    selected_bike = request.form.get('bike_name', 'Model')
    try:
        if category == 'laptop':
            # --- LAPTOP LOGIC ---
            form_values = request.form.to_dict()
            brand_enc = le_brand.transform([request.form.get('brand')])[0]
            cpu_brand_enc = le_cpu_brand.transform([request.form.get('cpu_brand')])[0]
            cpu_tier_enc = le_cpu_tier.transform([request.form.get('cpu_tier')])[0]
            ppi = ((float(request.form.get('res_width'))**2 + float(request.form.get('res_height'))**2)**0.5) / float(request.form.get('inches'))
            st_val = int(request.form.get('storage')) * 2 if request.form.get('is_ssd') == 'SSD' else int(request.form.get('storage'))
            
            # 4. Predict
            query = np.array([[brand_enc, int(request.form.get('ram', 8)), st_val, ppi, 
                               cpu_brand_enc, cpu_tier_enc, int(request.form.get('rating', 75))]])
            prediction = models['laptop'].predict(query)[0]
            final_price_string = f"₹{int(prediction):,}"

        elif category == 'mobile':
            # --- MOBILE LOGIC ---
            form_values = request.form.to_dict()
            query = np.array([[float(request.form.get('ram', 0)), float(request.form.get('camera', 0)), 
                               float(request.form.get('battery', 0)), float(request.form.get('screen', 0)), 
                               int(request.form.get('year', 2024)), int(request.form.get('brand_tier', 1)), 
                               int(request.form.get('processor_score', 0))]])
            prediction = models['mobile'].predict(query)[0]
            final_price_string=f"₹{int(prediction):,}"

        elif category == 'bike':
            # --- BIKE LOGIC ---
            form_values= request.form.to_dict()
            
            bike_data = pd.DataFrame([[
                float(request.form.get('Engine', 0)), 
                float(request.form.get('Mileage', 0)), 
                float(request.form.get('Max_Power', 0)), 
                float(request.form.get('Fuel_Capacity', 0)), 
                float(request.form.get('Top_Speed', 0)), 
                float(request.form.get('Kerb_Weight', 0)), 
                request.form.get('Tyre_Type', 'Tubeless'), 
                float(request.form.get('Rating', 7.5)), 
                float(request.form.get('Max_Torque', 0))
            ]], columns=['Engine', 'Mileage', 'Max_Power', 'Fuel_Capacity', 'Top_Speed', 'Kerb_Weight', 'Tyre_Type', 'Rating', 'Max_Torque'])
            
            # --- THE FIX: Convert 'Tyre_Type' to a category type ---
            bike_data['Tyre_Type'] = bike_data['Tyre_Type'].astype('category')
            prediction = models['bike'].predict(bike_data)[0]
            final_price_string= f"₹{int(prediction):,}"

        elif category == 'car':
            # --- CAR LOGIC ---
            # 1. Get data from form
            # Use .get(key, default) to prevent errors if a field is empty
            form_values= request.form.to_dict()
            car_df = pd.DataFrame([{
                'RTO': request.form.get('city_name', 'Unknown'),
                'Ownership': request.form.get('ownership', 'First Owner'),
                'Fuel Type': request.form.get('fuel', 'Petrol'),
                'Transmission': request.form.get('transmission', 'Manual'),
                'Insurance': request.form.get('insurance', 'Third Party'),
                'Year of Manufacture': int(request.form.get('year', 2020)),
                'Engine Displacement': request.form.get('engine', '1200'),
                'Kms Driven': request.form.get('kms', '10000'),
                'Seats': request.form.get('seats', '5')
            }])
            prediction = models['car'].predict(car_df)[0]
            final_price_string = format_car_price(prediction)

        
        
        template_map = {
        'laptop': 'index.html',
        'mobile': 'index2.html',
        'bike': 'index3.html',
        'car': 'index4.html'
        }
        #target_template = template_map.get(category, 'home.html')
        target_file = template_map.get(category, 'home.html')

        # CRITICAL: 'prediction_text' must match the variable name in your HTML
        return render_template(target_file, 
                                prediction_text=final_price_string, # Primary
                                result=final_price_string,          # Backup for Mobile
                                bike_name=selected_bike,            # Specifically for Bike page
                                category=category, 
                                form_data=form_values,
                                inputs=request.form,
                                **dropdown_data)
            
    #target_file, prediction_text=f"Estimated Price: {final_res}", category=category, **get_dropdown_data())

    except Exception as e:
# --- THE FIX: Return to target_file, NOT index.html ---
        error_msg = f"Prediction Error: {str(e)}"
        return render_template(target_file, 
                                prediction_text=error_msg, 
                                result=f"Error: {error_msg}", 
                                bike_name=selected_bike,
                                category=category, 
                                form_data=form_values,
                                inputs=request.form,
                                **dropdown_data)
    
    
if __name__ == '__main__':
    # Render provides a PORT environment variable, we must use it
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port,debug=True)
    