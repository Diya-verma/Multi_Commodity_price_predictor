from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load your model and the dataset (to get unique values for dropdowns)
model = pickle.load(open('bike_model.pkl', 'rb')) # Ensure this filename matches your dump file
df = pd.read_csv('bike_specifications.csv') # Used to populate dropdown lists

@app.route('/')
def index():
    # Get unique values for dropdowns to keep the UI dynamic
    bike_names = sorted(df['bike_name'].unique())
    tyre_types = sorted(df['Tyre_Type'].fillna('Unknown').unique().astype(str))
    
    return render_template('index3.html', 
                           bike_names=bike_names, 
                           tyre_types=tyre_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Extract data from form (Mapping HTML 'name' to variables)
        bike_name = request.form.get('bike_name')
        rating = float(request.form.get('Rating'))
        engine = float(request.form.get('Engine'))
        mileage = float(request.form.get('Mileage'))
        max_power = float(request.form.get('Max_Power'))
        fuel_cap = float(request.form.get('Fuel_Capacity'))
        top_speed = float(request.form.get('Top_Speed'))
        tyre_type = request.form.get('Tyre_Type')
        weight = float(request.form.get('Kerb_Weight'))
        torque = float(request.form.get('Max_Torque'))

        input_data = pd.DataFrame([[
                    engine,        # 1. Engine
                    mileage,       # 2. Mileage
                    max_power,     # 3. Max_Power
                    fuel_cap,      # 4. Fuel_Capacity
                    top_speed,     # 5. Top_Speed
                    weight,        # 6. Kerb_Weight
                    tyre_type,     # 7. Tyre_Type
                    rating,        # 8. Rating
                    torque         # 9. Max_Torque
                ]], columns=[
                                'Engine', 'Mileage', 'Max_Power', 'Fuel_Capacity',
                                'Top_Speed', 'Kerb_Weight', 'Tyre_Type', 'Rating', 'Max_Torque'
                    ])

        # --- CRITICAL FIX START ---
        # Convert 'object' (strings) to 'category' type
        
        input_data['Tyre_Type'] = input_data['Tyre_Type'].astype('category')
        
        # If your model is XGBoost, it needs this flag enabled
        # Note: If your model was saved as a Pipeline, it might not need this.
        # But for raw XGBoost models, use:
        prediction = model.predict(input_data) 
        # --- CRITICAL FIX END ---
        
        formatted_price = f"₹{int(prediction[0]):,}"

        # Reload dropdowns for the template
        bike_names = sorted(df['bike_name'].fillna('Unknown').unique().astype(str))
        tyre_types = sorted(df['Tyre_Type'].fillna('Unknown').unique().astype(str))
        
        return render_template('index3.html', 
                               prediction_text=formatted_price,
                               bike_name=bike_name,
                               bike_names=bike_names)

    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)