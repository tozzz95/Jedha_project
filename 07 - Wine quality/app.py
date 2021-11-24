from flask import Flask,request, url_for, redirect, render_template, jsonify
import requests
import pandas as pd
import pickle
import numpy as np
import joblib
app = Flask(__name__)
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)
#cols=['fixed_acidity','volatile_acidity' , 'citric_acid' ,
#'total_sulfur_dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol' ,
#'residual_sugar' , 'chlorides' , 'free_sulfur_dioxide']

@app.route("/documentation", methods=["GET", "POST"])
def docu():
    if request.method == "GET":
        return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def home_app():
    if request.method == 'GET':
        return(render_template("home_app.html"))

    if request.method == 'POST':
        fixed_acidity = request.form['fixed_acidity']
        volatile_acidity = request.form['volatile_acidity']
        citric_acid = request.form['citric_acid']
        total_sulfur_dioxide = request.form['total_sulfur_dioxide']
        density = request.form['density']
        pH = request.form['pH']
        sulphates = request.form['sulphates']
        alcohol = request.form['alcohol']
        residual_sugar = request.form['residual_sugar']
        chlorides = request.form['chlorides']
        free_sulfur_dioxide = request.form['free_sulfur_dioxide']

        pred = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid,
        total_sulfur_dioxide, density, pH, sulphates, alcohol,residual_sugar,
        chlorides, free_sulfur_dioxide]],
        columns=['fixed_acidity','volatile_acidity' , 'citric_acid' ,
        'total_sulfur_dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol' ,
        'residual_sugar' , 'chlorides' , 'free_sulfur_dioxide'], dtype = float)

        prediction = model.predict(pred)[0]
        
        return render_template("home_app.html", pred='{} / 10'.format(prediction))

#@app.route('/predict',methods=['POST'])
#def predict():
#    pred = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid,
#        total_sulfur_dioxide, density, pH, sulphates, alcohol,residual_sugar,
#        chlorides, free_sulfur_dioxide]],
#        columns=['fixed_acidity','volatile_acidity' , 'citric_acid' ,
#        'total_sulfur_dioxide' , 'density' , 'pH' , 'sulphates' , 'alcohol' ,
#        'residual_sugar' , 'chlorides' , 'free_sulfur_dioxide'], dtype = float)    
#    prediction = model.predict(pred)
#    prediction = float(prediction[0])
#    print("This is the prediction : ", prediction)
#    return jsonify({"prediction": prediction})

@app.route('/predict',methods=['POST'])
def predict():
    if request.json:
        req = request.get_json(force=True)

        if "input" in req.keys():
            classifier = joblib.load('model.joblib')
            prediction = classifier.predict(req["input"])
            print(prediction)
            prediction = float(prediction[0])
            return jsonify({"This is the prediction": prediction}), 200
        else:
            return jsonify({"msg": "Error: not a JSON or no specific key in your request"})
    return jsonify({"msg": "Error: not a JSON or no specific key in your request"})
    #print(req["input"])
    #prediction = model.predict(req["input"])
    #print(prediction)
    #output = [float(x) for x in prediction]
    #return jsonify({"predictions": output}),200

        #return render_template("home_app.html", pred='{} / 10'.format(prediction))
    
if __name__ == '__main__':
    app.run(debug=True)

