from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(f)) for f in [
        'Pleased', 'Satisfied', 'Beauty', 'Attractive',
        'Fit', 'Rewarding', 'HappyMemories', 'MentallyAlert'
    ]]
    prediction = model.predict([features])[0]
    label = "Happy" if prediction == 1 else "Unhappy"
    return render_template('index.html', prediction_text=label)


if __name__ == '__main__':
    app.run(debug=True)
