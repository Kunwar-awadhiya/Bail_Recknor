import numpy as np
import joblib
from flask import Flask, request, jsonify
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Load the trained model
model = joblib.load('bail_model.pkl')

# Flask app setup
app = Flask(__name__)

# Preprocess the offense type


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

# Prediction route


@app.route('/predict', methods=['POST'])
def predict_bail():
    try:
        data = request.get_json()

        # Preprocess the offense type
        offense_type = preprocess_text(data['offense_type'])

        # Convert case_date to datetime and create 'is_after_july2024' feature
        case_date = pd.to_datetime(data['case_date'])
        is_after_july2024 = int(case_date > pd.to_datetime('2024-07-01'))

        # Prepare input data for the model (ensure the input order matches training)
        input_data = {
            'robbery': 0, 'cybercrime': 0, 'fraud': 0, 'assault': 0, 'terrorism': 0,
            'economic offense': 0, 'crimes against women': 0, 'crimes against scs and sts': 0,
            'time_served': data['time_served'],
            'flight_risk': data['flight_risk'],
            'judicial_discretion': data['judicial_discretion'],
            'bailable_offense': data['bailable_offense'],
            'influence_on_witness': data['influence_on_witness'],
            'is_after_july2024': is_after_july2024
        }

        # Set the appropriate one-hot encoding for the offense_type
        if offense_type in input_data:
            input_data[offense_type] = 1

        # Convert the input_data to a DataFrame (since model expects it this way)
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Return the result
        return jsonify({
            'bail_granted': int(prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
