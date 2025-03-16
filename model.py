import pandas as pd
import numpy as np
import pickle
import re
import joblib
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Load dataset
df = pd.read_csv("job_descriptions.csv")

# Drop unnecessary columns
columns_to_drop = [
    "Job Id", "location", "Country", "latitude", "longitude", 
    "Job Posting Date", "Preference", "Contact Person", "Contact", 
    "Job Portal", "Company Profile"
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Function to extract numeric experience
def extract_experience(exp):
    if pd.isna(exp) or exp.lower() == "fresher":
        return 0  # Convert 'Fresher' to 0 years of experience
    numbers = re.findall(r'\d+', exp)
    return float(numbers[0]) if numbers else np.nan  # Take the first number found

# Apply conversion to 'Experience' column
df['Experience'] = df['Experience'].astype(str).apply(extract_experience).astype("float32")

# Apply Feature Hashing to the 'skills' column (Sparse Matrix)
vectorizer = HashingVectorizer(n_features=1000, alternate_sign=False)
skills_sparse = vectorizer.fit_transform(df['skills'].fillna(""))

# Drop 'skills' after vectorization
df.drop(columns=["skills"], inplace=True, errors='ignore')

# Apply Label Encoding to all string columns
label_encoders = {}  # Store encoders for later decoding

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to numerical labels
    label_encoders[col] = le  # Save encoder for future decoding

# Convert numerical DataFrame to a sparse format
df_sparse = np.array(df, dtype=np.float32)  # Convert to NumPy for efficiency
df_sparse = hstack([df_sparse, skills_sparse])  # Merge with sparse matrix

# Define features (X) and target (y)
X = df_sparse  # Features are now in sparse format
y = df[["Job Title", "Role"]]  # Targets (Multi-class)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifier
rf = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42)
multi_target_model = MultiOutputClassifier(rf)

# Train the model
multi_target_model.fit(X_train, y_train)

# Save model and encoders using pickle
# Save model separately
joblib.dump(multi_target_model, "career_model_only.pkl",compress=3)

# Save encoders separately
joblib.dump(label_encoders, "career_encoders.pkl",compress=2)
# Load and test predictions
# Load model separately
loaded_model = joblib.load("career_model_only.pkl")

# Load encoders separately
encoders = joblib.load("career_encoders.pkl")

# Make some predictions
sample_input = X_test[:5]  # Get 5 test samples
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
