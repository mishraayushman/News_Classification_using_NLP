import re
import nltk
import string
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

from src.News_Classification.exception import NewsClassificationException
from src.News_Classification.logger import logging
from dataclasses import dataclass
from src.News_Classification.components.data_ingestion import DataIngestionConfig,DataIngestion

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
from src.News_Classification.utils import save_object

@dataclass
class TextPreprocessorConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    transformed_Xtrain_file_path = os.path.join("transformed","Xtrain.npy")
    transformed_Xtest_file_path = os.path.join("transformed","Xtest.npy")
    y_train_file_path = os.path.join("transformed","y_train.npy")
    y_test_file_path = os.path.join("transformed","y_test.npy")

class TextPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.text_preprocessor_config = TextPreprocessorConfig()
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.column_transformer = None
        self.data_ingestion_config = DataIngestionConfig()
    
    
        
    def clean_text(self,text:str)->str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = BeautifulSoup(text, 'html.parser').get_text()
        words = [
            self.lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self.stop_words
        ]
        return " ".join(words)
       
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text)
            
class DataTransformation:
    def __init__(self):
        self.config = TextPreprocessorConfig()

    def get_preprocessor(self) -> ColumnTransformer:
        try:
            title_pipeline = Pipeline([
                ("cleaner",TextPreprocessor()),
                ("tfidf",TfidfVectorizer(max_features=2000,ngram_range=(2,2)))
            ])   

            content_pipeline = Pipeline([
                ("cleaner",TextPreprocessor()),
                ("tfidf",TfidfVectorizer(max_features=5000,ngram_range=(2,2)))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("title", title_pipeline, "title"),
                    ("content", content_pipeline, "content")
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise NewsClassificationException(e,sys)
    
    def initiate_data_transformation(self,train_path:str,test_path:str):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Splitting into features and target")
            target_column = "category"   # 👈 change this to your actual target column name
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logging.info("Encoding target feature")
            le = LabelEncoder()
            y_train_en = le.fit_transform(y_train)
            y_test_en = le.transform(y_test)
                        
            logging.info("Getting preprocessing object")
            preprocessor = self.get_preprocessor()

            logging.info("Fitting preprocessing object")
            X_train_tr= preprocessor.fit_transform(X_train).toarray()
            X_test_tr= preprocessor.transform(X_test).toarray()

            os.makedirs("transformed",exist_ok=True)
            
            logging.info("Saving numpy array")
            np.save(self.config.transformed_Xtrain_file_path,X_train_tr)
            np.save(self.config.transformed_Xtest_file_path,X_test_tr)
            np.save(self.config.y_train_file_path,y_train_en)
            np.save(self.config.y_test_file_path,y_test_en)


            logging.info("Saving Preprocessor object")
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                object=preprocessor
            )

            
            
            logging.info("Data Transformation Completed")
            return(
                self.config.transformed_Xtrain_file_path,
                self.config.transformed_Xtest_file_path,
                self.config.y_train_file_path,
                self.config.y_test_file_path
            )
            

        except Exception as e:
            raise NewsClassificationException(e,sys)
    


    


