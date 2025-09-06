import os
import sys
import pymysql
from src.News_Classification.exception import NewsClassificationException
from src.News_Classification.logger import logging
from dotenv import load_dotenv
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,accuracy_score
import dill
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
        df = pd.read_sql_query("select * from bbc_data",mydb)
        print(df.head())
        return df
    except Exception as e:
        raise NewsClassificationException(e,sys)

def save_object(file_path:str,object:object):
    try:
        logging.info("Entered the SaveObject block")
        with open(file_path,"wb") as file:
            pickle.dump(object,file)
        logging.info("Object saved")
    except Exception as e:
        raise NewsClassificationException(e,sys)
    
def eval_model(x_train,y_train,x_test,y_test,models:dict,params:dict):
    try:
        report = {}

        for i in range(len(list(models))):
            model =list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_score = accuracy_score(y_train,y_train_pred)
            test_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_score

            return report

    except Exception as e:
        raise NewsClassificationException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise NewsClassificationException(e, sys)