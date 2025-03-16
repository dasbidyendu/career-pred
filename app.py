from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer

app = Flask(__name__)
CORS(app)

# Load the trained model and encoders
loaded_model = joblib.load("career_model_only.pkl")
encoders = joblib.load("career_encoders.pkl")

# Reinitialize HashingVectorizer with the same settings used in training
vectorizer = HashingVectorizer(n_features=1006, alternate_sign=False)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get JSON input

        # Define required fields
        required_fields = ["Experience", "Qualifications", "Salary Range", "Work Type", "Company Size", "skills"]

        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Preprocess input
        processed_input = []

        # Encode categorical values safely
        for col in ["Qualifications", "Work Type", "Company Size"]:
            if col in encoders:
                if data[col] in encoders[col].classes_:
                    processed_input.append(encoders[col].transform([data[col]])[0])
                else:
                    processed_input.append(0)  # Default encoding for unseen labels
            else:
                processed_input.append(0)  # Default value if encoder is missing

        # Convert numerical values safely
        try:
            processed_input.append(float(data["Experience"]))
            processed_input.append(float(data["Salary Range"]))
        except ValueError:
            return jsonify({"error": "Invalid numerical input for Experience or Salary Range"}), 400

        # Vectorize skills (Sparse Matrix)
        skills_vector = vectorizer.transform([data["skills"]])

        # Convert processed_input to a NumPy array
        processed_input = np.array([processed_input], dtype=np.float32)

        # Combine numerical and sparse features
        final_input = hstack([processed_input, skills_vector])

        # Make predictions
        predictions = loaded_model.predict(final_input)

        # Decode predictions safely
        try:
            decoded_predictions = {
                "Job Title": encoders["Job Title"].inverse_transform([int(predictions[0][0])])[0],
                "Role": encoders["Role"].inverse_transform([int(predictions[0][1])])[0]
            }
        except Exception as decode_error:
            return jsonify({"error": f"Error decoding predictions: {str(decode_error)}"}), 500

        return jsonify(decoded_predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
