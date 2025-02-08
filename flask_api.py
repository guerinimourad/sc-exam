from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features
        features = np.array([[ 
            data["Age"], 
            data["Gender"], 
            data["Race/Ethnicity"], 
            data["BMI"], 
            data["SmokingStatus"], 
            data["TumorSize"] 
        ]])

        # Make prediction
        prediction = model.predict(features)
        result = "Yes" if prediction[0] == 1 else "No"

        # Return JSON response
        return jsonify({"Recurrence Prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
