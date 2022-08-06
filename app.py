
import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model_rf = pickle.load(open('randomforest1.pkl','rb')) 
model_gb = pickle.load(open('svm_model1.pkl','rb'))


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    
    '''
    For rendering results on HTML GUI
    '''
    pclass = float(request.args.get('pclass'))
    age = float(request.args.get('age'))
    sibsp = float(request.args.get('sibsp'))
    parch = float(request.args.get('parch'))
    sex = float(request.args.get('sex'))
    fare = float(request.args.get('fare'))
    
    

    
    
    if request.form.get('rf') == 'rf':
        prediction = model_rf.predict([[pclass,sex,age,sibsp,parch,fare]])
    else:
      prediction = model_gb.predict([[pclass,sex,age,sibsp,parch,fare]])
    if prediction==[1]:
      prediction_text="Model  has predicted that the person is Survived :{}".format(prediction)
    else:
      prediction_text="Model  has predicted that the person is Not Survived :{}".format(prediction)
        
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run()
