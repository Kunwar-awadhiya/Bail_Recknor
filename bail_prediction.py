import pandas as pd
import numpy as np
import joblib
from tkinter import Tk, Label, Entry, Button, StringVar, IntVar, messagebox, OptionMenu
from tkcalendar import DateEntry
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz  # Fuzzy matching
from fuzzywuzzy import process  # For fuzzy matching suggestions

# Load the trained models
bail_model = joblib.load('bail_model.pkl')
legal_model = joblib.load('legal_explanation_model.pkl')

# Load the legal data
legal_data = pd.read_csv('legal_data.csv')

# Expanded offense types based on provided data
offense_types = [
    'assault', 'crimes against scs and sts', 'crimes against women',
    'cybercrime', 'domestic violence', 'drug offense', 'economic offense',
    'fraud', 'robbery', 'sexual assault', 'terrorism', 'theft', 'vandalism'
]


def preprocess_text(text):
    """Tokenize and lower-case the text."""
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)


def get_legal_suggestion(offense_type):
    """Fetch legal suggestions based on the offense type using fuzzy matching."""
    # Preprocess the input
    offense_type = preprocess_text(offense_type)
    print(f"Processed Offense Type: {offense_type}")  # Debugging print

    # Perform fuzzy matching to find the closest match in the legal data
    closest_match = process.extractOne(
        offense_type, legal_data['offense_type'].str.lower(), scorer=fuzz.token_sort_ratio)

    # Check fuzzy match score and ensure it's valid
    if closest_match and closest_match[1] > 70:
        matched_offense = closest_match[0]
        suggestion = legal_model.predict([matched_offense])[0]
        return suggestion
    else:
        print(f"No close match found for: {offense_type}")  # Debugging print
        return "No legal suggestion available for this offense type."


def predict_bail():
    """Predict bail based on user inputs and display all inputs with corresponding values."""
    # Get inputs from the GUI
    offense_type = offense_type_var.get()
    ipc_section = ipc_section_var.get()  # Not used in prediction, just for display
    time_served = int(time_served_var.get())
    bailable_offense = int(bailable_offense_var.get())
    compoundable = int(compoundable_var.get())
    flight_risk = int(flight_risk_var.get())
    judicial_discretion = int(judicial_discretion_var.get())
    influence_on_witness = int(influence_on_witness_var.get())
    previous_criminal_record = int(previous_criminal_record_var.get())
    case_date = pd.to_datetime(case_date_var.get())

    # Check if the case date is after July 1, 2024
    is_after_july2024 = int(case_date > pd.to_datetime('2024-07-01'))

    # Preprocess and encode offense_type
    offense_type = preprocess_text(offense_type)

    # Fuzzy matching to get the best match for offense type
    legal_suggestion = get_legal_suggestion(offense_type)

    # Encode offense type based on the provided categories
    offense_type_encoded = [int(offense_type == ot) for ot in offense_types]

    # Prepare input data with exactly 19 features
    input_data = np.array([[
        *offense_type_encoded,  # Encoded offense type (13 categories)
        time_served,            # Time served in months
        flight_risk,            # Flight risk (1-10)
        judicial_discretion,    # Judicial discretion (1-10)
        bailable_offense,       # Bailable offense (0 or 1)
        influence_on_witness,   # Influence on witness (1-10)
        is_after_july2024       # Is case after July 1, 2024?
    ]])

    # Ensure input data shape is correct
    if input_data.shape[1] != 19:
        raise ValueError(
            f"Input data shape mismatch! Expected 19 features, but got {input_data.shape[1]}.")

    # Make prediction
    try:
        prediction = bail_model.predict(input_data)[0]
        result = "Bail Granted" if prediction == 1 else "Bail Denied"

        # Format user inputs for display
        user_inputs = f"""
        **User Inputs:**
        - Offense Type: {offense_type}
        - IPC Section: {ipc_section}
        - Time Served: {time_served} months
        - Bailable Offense: {"Yes" if bailable_offense == 1 else "No"}
        - Compoundable: {"Yes" if compoundable == 1 else "No"}
        - Flight Risk: {flight_risk}/10
        - Judicial Discretion: {judicial_discretion}/10
        - Influence on Witness: {influence_on_witness}/10
        - Previous Criminal Record: {"Yes" if previous_criminal_record == 1 else "No"}
        - Case Date: {case_date.date()}
        - Is After July 1, 2024: {"Yes" if is_after_july2024 == 1 else "No"}

        **Prediction Result:**
        {result}

        **Legal Suggestion:**
        {legal_suggestion}
        """

        # Display message if the case date is after July 1, 2024
        if is_after_july2024 == 1:
            user_inputs += "\n\n**Note:** Laws passed after July 1, 2024, will be used for this case."

    except ValueError as e:
        print(f"Prediction error: {e}")
        user_inputs = f"Error in prediction: {e}"

    # Display result with user inputs
    messagebox.showinfo("Prediction Result", user_inputs)



def explain_input():
    explanations = """
    Input Explanations:
    - Offense Type: Type of offense (e.g., Robbery, Cybercrime, Fraud, etc.).
    - IPC Section: Relevant section of the Indian Penal Code (IPC).
    - Time Served: Number of months the person has already served.
    - Bailable Offense: Whether the offense is bailable (0 = No, 1 = Yes).
    - Compoundable: Whether the offense is compoundable (0 = No, 1 = Yes).
    - Flight Risk: Risk of fleeing the jurisdiction (1-10 scale).
    - Judicial Discretion: Level of discretion available to the judge (1-10 scale).
    - Influence on Witness: Potential influence on witnesses (1-10 scale).
    - Previous Criminal Record: Whether the person has a previous criminal record (0 = No, 1 = Yes).
    - Case Date: The date of the case.
    - Is After July 2024: Whether the case date is after July 1, 2024 (0 = No, 1 = Yes).
    """
    messagebox.showinfo("Input Explanations", explanations)


# Set up GUI
root = Tk()
root.title("Bail Prediction System")

# Define variables
offense_type_var = StringVar()
ipc_section_var = StringVar()
time_served_var = StringVar()
bailable_offense_var = IntVar()
compoundable_var = IntVar()
flight_risk_var = IntVar()
judicial_discretion_var = IntVar()
influence_on_witness_var = IntVar()
previous_criminal_record_var = IntVar()
case_date_var = StringVar()

# Create GUI elements
Label(root, text="Offense Type").grid(row=0, column=0)
OptionMenu(root, offense_type_var, *offense_types).grid(row=0, column=1)

Label(root, text="IPC Section").grid(row=1, column=0)
Entry(root, textvariable=ipc_section_var).grid(row=1, column=1)

Label(root, text="Time Served (months)").grid(row=2, column=0)
Entry(root, textvariable=time_served_var).grid(row=2, column=1)

Label(root, text="Bailable Offense (0 for NO or 1)").grid(row=3, column=0)
Entry(root, textvariable=bailable_offense_var).grid(row=3, column=1)

Label(root, text="Compoundable (0 or 1)").grid(row=4, column=0)
Entry(root, textvariable=compoundable_var).grid(row=4, column=1)

Label(root, text="Flight Risk (1-10)").grid(row=5, column=0)
Entry(root, textvariable=flight_risk_var).grid(row=5, column=1)

Label(root, text="Judicial Discretion (1-10)").grid(row=6, column=0)
Entry(root, textvariable=judicial_discretion_var).grid(row=6, column=1)

Label(root, text="Influence on Witness (1-10)").grid(row=7, column=0)
Entry(root, textvariable=influence_on_witness_var).grid(row=7, column=1)

Label(root, text="Previous Criminal Record(0 or 1)").grid(row=8, column=0)
Entry(root, textvariable=previous_criminal_record_var).grid(row=8, column=1)

Label(root, text="Case Date").grid(row=9, column=0)
DateEntry(root, textvariable=case_date_var).grid(row=9, column=1)

Button(root, text="Predict Bail", command=predict_bail).grid(row=10, column=0)
Button(root, text="Explain Input", command=explain_input).grid(row=10, column=1)

# Run the GUI event loop
root.mainloop()
