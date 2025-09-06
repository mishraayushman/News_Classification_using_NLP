import sys
import os 
import pandas as pd
import numpy as np
from dataclasses import dataclass
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score,mean_squared_error

from src.News_Classification.exception import NewsClassificationException
from src.News_Classification.logger import logging
from src.News_Classification.utils import eval_model,save_object

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,x_train,y_train,x_test,y_test):
        try:
            logging.info("models Trainer initializing")
            models = {
                "Random Forest":RandomForestClassifier(),
                "Logistic Regression":LogisticRegression(),
                "Decision Tree" : DecisionTreeClassifier(),
                "SVC": SVC(),
                "KNN":KNeighborsClassifier()
            }
            param_grids = {
            "Random Forest": {
                # "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
                },

            "Logistic Regression": {
                "penalty": ["l1", "l2", "elasticnet", None],
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "saga"],
                # "max_iter": [100, 500, 1000]
            },

            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "splitter": ["best", "random"]
            },

            "SVC": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": [2, 3, 4],
                "gamma": ["scale", "auto"]
            },

                "KNN": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                }
            }

            model_report:dict = eval_model(x_train=x_train,y_train=y_train,x_test=x_test
                                           ,y_test=y_test,models=models,params=param_grids)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise NewsClassificationException("no best models",sys)
            logging.info("best models found on both train and test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )

            return best_model,best_model_score

        except Exception as e:
            raise NewsClassificationException(e,sys)