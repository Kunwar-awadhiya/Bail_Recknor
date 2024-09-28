import pandas as pd
import numpy as np
import nltk
import joblib
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Download necessary NLTK resources
nltk.download('punkt')

# Example dataset (this needs to be replaced with actual legal data)
data = {
    'offense_type': [
        'Robbery', 'Cybercrime', 'Fraud', 'Assault', 'Terrorism',
        'Economic Offense', 'Crimes Against Women', 'Crimes Against SCs and STs',
        'Theft', 'Drug Offense', 'Vandalism', 'Domestic Violence', 'Sexual Assault'
    ],
    'ipc_section': [
        'IPC 392', 'IT Act 66', 'IPC 420', 'IPC 323', 'UAPA Sec 16',
        'IPC 406', 'IPC 498A', 'SC/ST Act Sec 3(1)(r)', 'IPC 378', 'NDPS Act Sec 20',
        'IPC 427', 'IPC 498A', 'IPC 376'
    ],
    'time_served': [12, 8, 5, 1, 20, 10, 4, 9, 3, 6, 2, 15, 14],  # In months
    'bailable_offense': [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    'compoundable': [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    'flight_risk': [5, 3, 4, 2, 9, 6, 3, 7, 4, 8, 2, 7, 5],  # Scale of 1 to 10
    'judicial_discretion': [8, 6, 7, 5, 9, 6, 4, 7, 8, 3, 5, 6, 2],
    'influence_on_witness': [6, 4, 3, 2, 9, 5, 7, 8, 5, 3, 1, 4, 6],
    'previous_criminal_record': [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
    'case_date': [
        '2024-05-15', '2024-06-02', '2024-05-19', '2024-07-10',
        '2023-12-05', '2024-01-20', '2023-07-15', '2024-08-20',
        '2024-06-10', '2024-05-25', '2024-07-05', '2023-11-30',
        '2024-04-15'
    ],
    'bail_granted': ['No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No']
}

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(data)

# Convert the case_date column to datetime
df['case_date'] = pd.to_datetime(df['case_date'])

# Add a feature that checks if the case date is after July 1, 2024
df['is_after_july2024'] = df['case_date'] > pd.to_datetime('2024-07-01')

# Preprocess text: Tokenize and clean the offense type


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)


df['offense_type'] = df['offense_type'].apply(preprocess_text)

# Dummy encoding for offense_type (13 categories)
df_encoded = pd.get_dummies(df['offense_type'])

# Convert bail_granted to numeric labels
df['bail_granted'] = df['bail_granted'].map({'No': 0, 'Yes': 1})

# Combine the encoded offense_type with other numerical features, including the new date-based feature
X = pd.concat([df_encoded, df[['time_served', 'flight_risk', 'judicial_discretion','bailable_offense', 'influence_on_witness', 'is_after_july2024']]], axis=1)
y = df['bail_granted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the model
model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'bail_model.pkl')
print("Model saved as 'bail_model.pkl'")

# Expanded Legal Context and Suggestions
# Expanded Legal Context and Suggestions
