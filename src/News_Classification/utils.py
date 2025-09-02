import os
import sys
import pymysql
from src.News_Classification.exception import NewsClassificationException
from src.News_Classification.logger import logging
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')
# print(host,user,password,db)
def read_sql_data():
    try:
        logging.info(f"connecting to mysql:{db}")
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established to: {mydb}")
        df = pd.read_sql_query("select * from bbc_news",mydb)
        print(df.head())
        return df
    except Exception as e:
        raise NewsClassificationException(e,sys)