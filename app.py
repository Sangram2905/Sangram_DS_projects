import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('psp_Sequential_model.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def SalesInput():
    return render_template('SalesInput.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("SalesInput.html",prediction_text="The Sales target prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)