from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            int(request.form['Gender']),
            int(request.form['Married']),
            int(request.form['Dependents']),
            int(request.form['Education']),
            int(request.form['Self_Employed']),
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            int(request.form['Property_Area'])
        ]

        arr = np.array(data).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        prediction = model.predict(arr_scaled)[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)
