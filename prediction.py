import pickle
import numpy as np

# Load the trained model and encoders
with open("career_model.pkl", "rb") as f:
    loaded_data = pickle.load(f)

loaded_model = loaded_data["model"]
encoders = loaded_data["encoders"]


sample_input = {
    "Experience": 3,  # Example: 3 years
    "Qualifications": "B.Tech",  # Example qualification
    "Salary Range": 60000,  # Example salary
    "Work Type": "Full Time",
    "Company Size": "Medium",
    "skills": "Python, Machine Learning, Data Analysis"
}
# Assuming X_test is already preprocessed and available
# Get 5 test samples

# Make predictions
predictions = loaded_model.predict(sample_input)

# Decode predictions back to original labels
decoded_predictions = []
for pred in predictions:
    decoded_pred = {
        "Job Title": encoders["Job Title"].inverse_transform([pred[0]])[0],
        "Role": encoders["Role"].inverse_transform([pred[1]])[0]
    }
    decoded_predictions.append(decoded_pred)

# Print predictions
for i, pred in enumerate(decoded_predictions):
    print(f"Sample {i+1}: Predicted Job Title: {pred['Job Title']}, Role: {pred['Role']}")
