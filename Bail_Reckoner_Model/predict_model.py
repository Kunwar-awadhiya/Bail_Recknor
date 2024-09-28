import numpy as np
import joblib
from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Load the trained model
model = joblib.load('./bail_model.pkl')

# Flask app to create API
app = Flask(__name__)

# Preprocess offense type


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

# Function to generate explanation and suggestions


def generate_reason_and_suggestions(data, prediction):
    reasons = []
    suggestions = []

    # Offense type analysis
    if 'cybercrime' in data['offense_type'].lower():
        reasons.append(
            "The offense is categorized as cybercrime, which might lead to stricter bail conditions.")
        suggestions.append(
            "Ensure you can provide a strong legal argument for bail in cases involving cybercrimes.")

    # Time served analysis
    if data['time_served'] < 6:
        reasons.append(
            "The time served is less than 6 months, which could reduce eligibility for bail.")
        suggestions.append(
            "Serve more time in custody to increase chances of eligibility.")
    else:
        reasons.append(
            f"The time served of {data['time_served']} months meets the eligibility requirements for bail.")

    # Flight risk analysis
    if data['flight_risk'] > 5:
        reasons.append(
            "The flight risk is high, which makes the court less likely to grant bail.")
        suggestions.append(
            "Provide stronger surety bonds to reduce concerns of flight risk.")
    else:
        reasons.append(
            "The flight risk is manageable, which supports bail eligibility.")

    # Judicial discretion analysis
    if data['judicial_discretion'] < 5:
        reasons.append(
            "Judicial discretion in this case is low, which could work against bail.")
        suggestions.append(
            "Highlight good behavior and lack of influence over the case to improve judicial discretion.")
    else:
        reasons.append(
            "Judicial discretion is high, which favors the granting of bail.")

    # Final result based on prediction
    if prediction == 1:
        result = "Eligible for Bail"
    else:
        result = "Not Eligible for Bail"

    return result, reasons, suggestions


@app.route('/predict', methods=['POST'])
def predict_bail():
    data = request.get_json()

    # Preprocess the offense type
    #offense_type = preprocess_text(data['offense_type'])

    # Inputs for the model
    input_data = np.array([[
        data['time_served'],
        data['flight_risk'],
        data['judicial_discretion']
    ]])

    # Load the model and predict
    prediction = model.predict(input_data)[0]

    # Generate reasoning and suggestions
    result, reasons, suggestions = generate_reason_and_suggestions(
        data, prediction)

    # Return detailed output
    return jsonify({
        'result': result,
        'reasons': reasons,
        'suggestions': suggestions
    })


if __name__ == '__main__':
    app.run(debug=True)
