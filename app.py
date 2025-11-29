import streamlit as st
import numpy as np
import pandas as pd

from src.News_Classification.pipelines.prediction_pipeline import PredictPipeline, CustomData
from sklearn.preprocessing import StandardScaler

# Streamlit App

st.title("News Classification App")

st.write("Provide the news title and content to classify the category.")

# Input fields (replacing Flask HTML form)
title = st.text_input("News Title")
content = st.text_area("News Content")

# Predict button
if st.button("Predict"):

    # Create data object same as Flask
    data = CustomData(
        title=title,
        content=content
    )

    pred_df = data.get_data_as_df()

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    st.success(f"Predicted Category: {results[0]}")
