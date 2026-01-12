from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            int(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            int(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]

        data = np.array(features).reshape(1, -1)
        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)

        return render_template(
            'index.html',
            prediction_text=f'Estimated House Price: ${round(prediction[0], 2)}'
        )

    except Exception as e:
        return render_template('index.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)
