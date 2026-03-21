from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 1. Saari files load karein
model = pickle.load(open('laptop_model.pkl', 'rb'))
le_brand = pickle.load(open('brand_encoder.pkl', 'rb'))
le_cpu_brand = pickle.load(open('cpu_brand_encoder.pkl', 'rb'))
le_cpu_tier = pickle.load(open('cpu_tier_encoder.pkl', 'rb'))

@app.route('/')
def home():
    # Brand aur CPU list ko HTML dropdown mein bhejne ke liye
    return render_template('index.html', 
                           brands=le_brand.classes_, 
                           cpu_brands=le_cpu_brand.classes_,
                           cpu_tiers=le_cpu_tier.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    # HTML Form se data nikalna
    brand = request.form.get('brand')
    ram = int(request.form.get('ram'))
    cpu_brand = request.form.get('cpu_brand')
    cpu_tier = request.form.get('cpu_tier')
    storage = int(request.form.get('storage'))
    is_ssd = request.form.get('is_ssd')
    inches = float(request.form.get('inches', 15.6))
    res_width = int(request.form.get('res_width', 1920))
    res_height = int(request.form.get('res_height', 1080))
    rating = int(request.form.get('rating', 70))

    # Piche ka logic (PPI aur Storage Score)
    ppi = ((res_width**2 + res_height**2)**0.5) / inches
    st_val = storage * 2 if is_ssd == 'SSD' else storage

    # Encoders use karke text ko number mein badalna
    brand_enc = le_brand.transform([brand])[0]
    cpu_brand_enc = le_cpu_brand.transform([cpu_brand])[0]
    cpu_tier_enc = le_cpu_tier.transform([cpu_tier])[0]

    # Prediction
    query = np.array([[brand_enc, ram, st_val, ppi, cpu_brand_enc, cpu_tier_enc, rating]])
    prediction = model.predict(query)[0]

    return render_template('index.html', 
                           prediction_text=f'Estimated Price: ₹{int(prediction):,}',
                           brands=le_brand.classes_, 
                           cpu_brands=le_cpu_brand.classes_,
                           cpu_tiers=le_cpu_tier.classes_)

if __name__ == "__main__":
    app.run(debug=True)