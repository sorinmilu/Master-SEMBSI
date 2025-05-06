# Define rules and knowledge base
RULES = [
    {
        "symptoms": ["fever", "fatigue", "cough", "body_ache"],
        "disease": "flu"
    },
    {
        "symptoms": ["fever", "fatigue", "cough", "shortness_of_breath"],
        "disease": "covid"
    },
    {
        "symptoms": ["headache", "fatigue", "nausea"],
        "disease": "migraine"
    },
    {
        "symptoms": ["fever", "chills", "body_ache", "fatigue"],
        "disease": "malaria"
    },
    {
        "symptoms": ["headache", "sensitive_to_light", "nausea"],
        "disease": "migraine"
    },
    {
        "symptoms": ["sore_throat", "runny_nose", "cough"],
        "disease": "common cold"
    }
]

def diagnose(symptoms):
    """Diagnoses possible diseases based on symptoms and rules."""
    possible_diseases = set()
    
    # Apply rules to symptoms
    for rule in RULES:
        if set(rule["symptoms"]).issubset(symptoms):
            possible_diseases.add(rule["disease"])
    
    if possible_diseases:
        return possible_diseases
    else:
        return "Sorry, I couldn't diagnose based on the symptoms."

def run(symptoms):
    """Main function to run the medical expert system."""
    # Diagnose diseases
    diseases = diagnose(symptoms)
    
    # Display results
    if isinstance(diseases, set):
        print("\nBased on the provided symptoms, you may have one of the following diseases:")
        for disease in diseases:
            print(f"- {disease}")
    else:
        print(f"\nDiagnosis: {diseases}")

# Main program
if __name__ == "__main__":
    # Example predefined list of symptoms
    predefined_symptoms = ["fever", "fatigue", "cough", "body_ache"]
    run(predefined_symptoms)
