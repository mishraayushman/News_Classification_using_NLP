import streamlit as st
import numpy as np
import pandas as pd

from src.News_Classification.pipelines.prediction_pipeline import PredictPipeline, CustomData
from sklearn.preprocessing import StandardScaler

# LABEL MAPPING
label_map = {
    0: "business",
    1: "entertainment",
    2: "politics",
    3: "sports",
    4: "tech"
}

st.title("News Classification App")

st.write("Provide the news title and content to classify the category.")

# Inputs
title = st.text_input("News Title")
content = st.text_area("News Content")

# Predict button
if st.button("Predict"):

    # Create input data
    data = CustomData(
        title=title,
        content=content
    )

    pred_df = data.get_data_as_df()

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    # Convert numeric prediction to label
    predicted_class_num = int(results[0])
    predicted_label = label_map[predicted_class_num]

    st.success(f"Predicted Category: **{predicted_label}**")
