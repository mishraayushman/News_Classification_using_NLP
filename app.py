from src.News_Classification.components.data_ingestion import DataIngestionConfig,DataIngestion
from src.News_Classification.logger import logging
from src.News_Classification.exception import NewsClassificationException
import sys
if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise NewsClassificationException(e,sys)