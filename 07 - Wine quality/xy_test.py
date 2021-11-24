from flask import Flask,request, url_for, redirect, render_template, jsonify
import requests
import pandas as pd
import pickle
import numpy as np
import joblib
app = Flask(__name__)
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)
cols=['fixed_acidity','volatile_acidity' , 'citric_acid' ,
'total_sulfur_dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol' ,
'residual_sugar' , 'chlorides' , 'free_sulfur_dioxide']

@app.route('/')
def home_app():
    return render_template("home_app.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_f = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_f)
    prediction = float(prediction[0])
    print("This is the prediction : ", prediction)
    return render_template('home_app.html', pred='The quality of this wine is : {} / 10'.format(prediction))

#@app.route('/predict_api',methods=['POST'])
#def predict_api():
#    req = request.get_json(force=True)
#    #data_unseen = pd.DataFrame([data])
#    print(req["input"])
#    prediction = model.predict(req["input"])
#    print(prediction)
#    output = [float(x) for x in prediction]
#    return jsonify({"predictions": output}),200

    
if __name__ == '__main__':
    app.run(debug=True)