from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

# Import our linear regression model and standard scaler joblib

linear_reg_model = joblib.load('models/fwi_prediction_model/linear_reg_model_predict_fwi.joblib')
standard_scaler = joblib.load('models/fwi_prediction_model/scaler_for_linear_reg_model_predict_fwi.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # 1. Capture Inputs
            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            Ws = float(request.form['Ws'])
            Rain = float(request.form['Rain'])
            FFMC = float(request.form['FFMC'])
            DMC = float(request.form['DMC'])
            ISI = float(request.form['ISI'])
            Classes = float(request.form['Classes'])
            Region = float(request.form['Region'])

            # 2. Scale
            # Ensure this list order matches your X_train columns EXACTLY
            new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            new_scaled_data = standard_scaler.transform(new_data)

            # 3. Predict
            result = linear_reg_model.predict(new_scaled_data)

            # 4. Return
            return render_template('home.html', results=result[0])

        except Exception as e:
            # This prints the error to your terminal so you can debug
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Error: Check inputs")
    
    else:
        # GET request (just showing the form)
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')

# if __name__ == "__main__":
#     main()
