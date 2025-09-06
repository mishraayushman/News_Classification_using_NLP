from src.News_Classification.components.data_ingestion import DataIngestionConfig,DataIngestion
from src.News_Classification.logger import logging
from src.News_Classification.exception import NewsClassificationException
from src.News_Classification.components.data_transformation import DataTransformation,TextPreprocessor,TextPreprocessorConfig
from src.News_Classification.components.model_tranier import ModelTrainer,ModelTrainerConfig
import sys
import numpy as np

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        data_ingestion_config = DataIngestionConfig()
        # data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        # data_transformation.initiate_data_transformation(data_ingestion_config.train_data_path,data_ingestion_config.test_data_path)
        ###
        model_trainer = ModelTrainer()
        x_train = np.load("transformed/Xtrain.npy")
        y_train = np.load("transformed/y_train.npy")
        x_test = np.load("transformed/Xtest.npy")
        y_test = np.load("transformed/y_test.npy")
        print(
            model_trainer.initiate_model_trainer(x_train=x_train,y_train=y_train
                                             ,x_test=x_test,y_test=y_test)
        )
    except Exception as e:
        raise NewsClassificationException(e,sys)