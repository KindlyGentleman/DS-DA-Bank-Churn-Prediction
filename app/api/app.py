import pickle
from flask import Flask, request, app, jsonify,url_for,render_template
import numpy as np
import pandas as pd
from utilities import prediction_pipeline

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    data_new = pd.DataFrame(data, index = [0])
    output= prediction_pipeline(data_new)
    if output[0] == 1:
        return jsonify("Churn")
    else:
        return jsonify("No Churn")

if __name__ == "__main__": 
    app.run(host="0.0.0.0",debug=True)