import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application # because we need to create the route later


## Route for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
                gender = request.form.get("gender"),
                race_ethnicity= request.form.get("race_ethnicity"),
                parental_level_of_education= request.form.get("parental_level_of_education"),
                lunch= request.form.get("lunch"),
                test_preparation_course= request.form.get("test_preparation_course"),
                reading_score= request.form.get("reading_score"),
                writing_score= request.form.get("writing_score"),
                )
        df_to_predict = data.get_data_as_frame()
        print(df_to_predict)
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(df_to_predict)
        return render_template('home.html',results="{:.2f}".format(predictions[0]))
    
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)