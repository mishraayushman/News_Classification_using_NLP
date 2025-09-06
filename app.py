from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from src.News_Classification.pipelines.prediction_pipeline import PredictPipeline,CustomData


from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])

def classify():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            title = request.form.get("title"),
            content = request.form.get("content")#127.0.0.1:5000
        )
        pred_df = data.get_data_as_df()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template("home.html",results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)