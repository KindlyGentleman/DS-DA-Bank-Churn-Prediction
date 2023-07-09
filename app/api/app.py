# Import necessary libraries
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
from utilities import prediction_pipeline

# Create Flask app instance
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    # Render home page template
    return render_template("home.html")

# Define route for prediction API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get data from request JSON
    data = request.json['data']
    # Convert data to DataFrame
    data_new = pd.DataFrame(data, index=[0])
    # Run prediction pipeline on data
    output = prediction_pipeline(data_new)
    # Return prediction result as JSON response
    if output[0] == 1:
        return jsonify("Churn")
    else:
        return jsonify("No Churn")

# Run Flask app if script is run directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
