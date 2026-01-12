from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))
print('model loaded')

# Load scaler
scaled = pickle.load(open('scaling.pkl', 'rb'))
print('scaler loaded')

# Load label encoder
wine_type_en = pickle.load(open('wine_type.pkl', 'rb'))
print('label encoder loaded')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form inputs (same order as training data)
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            print(fixed_acidity, volatile_acidity, citric_acid, residual_sugar)

            # Prepare data exactly like car app
            details = [
                fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                pH,
                sulphates,
                alcohol
            ]

            data_out = np.array(details).reshape(1, -1)
            print(data_out)
            print(data_out.shape)

            # Scale input
            data_scaled = scaled.transform(data_out)

            # Predict
            prediction = model.predict(data_scaled)
            print(prediction)

            # Decode output
            wine_type = wine_type_en.inverse_transform(
                [int(round(prediction[0]))]
            )[0]

            return render_template(
                'index.html',
                prediction_text=f'Predicted Wine Type: {wine_type}'
            )

        except Exception as e:
            return render_template(
                'index.html',
                prediction_text=f'Error: {str(e)}'
            )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
