import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# --- Yeh magic line hai jo file ka sahi rasta dhoondti hai ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'mobile_price_model.pkl')

# Model load karo
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print(f"Bhai, error! Model file is path par nahi mili: {model_path}")

@app.route('/')
def index():
    # Kyunki aapne kaha index2.html hai
    return render_template('index2.html')

# app2.py mein ye ensure karein:
@app.route('/predict', methods=['POST'])
def predict():
    # index2.html ke 'name' se match karein
    ram = float(request.form.get('ram'))
    camera = float(request.form.get('camera'))
    battery = float(request.form.get('battery'))
    screen = float(request.form.get('screen'))
    year = int(request.form.get('year'))
    brand_tier = int(request.form.get('brand_tier'))
    processor_score = int(request.form.get('processor_score'))

    features = np.array([[ram, camera, battery, screen, year, brand_tier, processor_score]])
    prediction = model.predict(features)[0]
    
    # Ye result string wahi hai jo HTML mein split ho rahi hai
    return render_template('index2.html', result=f"Estimated Price: ₹{round(prediction, 2)}")

if __name__ == '__main__':
    app.run(debug=True)