import sys
import pandas as pd

from src.News_Classification.exception import NewsClassificationException
from src.News_Classification.logger import logging

from src.News_Classification.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path="Artifacts\model.pkl"
            preprocessor_path = "Artifacts\preprocessor.pkl"
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_transformed = preprocessor.transform(features)
            pred = model.predict(data_transformed)
            return pred
        except Exception as e:
            raise NewsClassificationException(e,sys)

class CustomData:
    def __init__(self, title:str,content:str):
        self.title = title
        self.content = content
    
    def get_data_as_df(self):
        try:
            data_input = {
                "title":[self.title],
                "content":[self.content]
            }
            return pd.DataFrame(data_input)
        except Exception as e:
            raise NewsClassificationException(e,sys)