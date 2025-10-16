import pickle   
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for
from sklearn.preprocessing import StandardScaler

# Initialize Flask application
application = Flask(__name__)
app = application

# Load the scaler and model
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    """Route for the landing page"""
    return render_template('index.html')

@app.route('/home')
def home():
    """Route for the home page"""
    return render_template('home.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    """Route for making predictions"""
    if request.method == 'POST':
        try:
            # Get form data
            Hours_Studied = float(request.form.get('Hours_Studied'))
            Previous_Scores = float(request.form.get('Previous_Scores'))
            Extracurricular_Activities = float(request.form.get('Extracurricular_Activities'))
            Sleep_Hours = float(request.form.get('Sleep_Hours'))
            Sample_Question_Papers_Practiced = float(request.form.get('Sample_Question_Papers_Practiced'))

            # Validate input ranges
            if not (0 <= Hours_Studied <= 24 and
                   0 <= Previous_Scores <= 100 and
                   0 <= Extracurricular_Activities <= 40 and
                   0 <= Sleep_Hours <= 12 and
                   0 <= Sample_Question_Papers_Practiced <= 100):
                return render_template('home.html', error="Please enter values within the valid ranges.")

            # Scale the input features
            input_features = np.array([[
                Hours_Studied,
                Previous_Scores,
                Extracurricular_Activities,
                Sleep_Hours,
                Sample_Question_Papers_Practiced
            ]])
            scaled_features = scaler.transform(input_features)

            # Make prediction
            prediction = model.predict(scaled_features)
            
            # Round the prediction to 2 decimal places
            formatted_prediction = round(float(prediction[0]), 2)

            return render_template('home.html',
                                predictions=formatted_prediction,
                                input_data={
                                    'Hours_Studied': Hours_Studied,
                                    'Previous_Scores': Previous_Scores,
                                    'Extracurricular_Activities': Extracurricular_Activities,
                                    'Sleep_Hours': Sleep_Hours,
                                    'Sample_Question_Papers_Practiced': Sample_Question_Papers_Practiced
                                })

        except ValueError as e:
            return render_template('home.html', error="Please enter valid numeric values for all fields.")
        except Exception as e:
            return render_template('home.html', error="An error occurred while processing your request.")

    # GET request - just display the form
    return render_template('home.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    return render_template('index.html', error="An internal server error occurred."), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
           