import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression,ElasticNet,SGDRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from src.components.model_monitering import mlflow_logging

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "ElasticNet": ElasticNet(),
                "SGDRegressor": SGDRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor()
                
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{
                },
                "SGDRegressor":{
                    'loss' : ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'penalty' : ['l2', 'l1', 'elasticnet'],
                    'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    'learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
                    'eta0' : [0.01, 0.1, 0.5, 1, 10],
                    'max_iter' : [1000, 5000, 10000, 50000, 100000],
                    'tol' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
                },
                "GradientBoostingRegressor":{
                    "loss" : ["squared_error", "absolute_error", "huber", "quantile"],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsRegressor":{
                    'n_neighbors':[3,5,7,9,11],
                    'weights':['uniform','distance']
                },
                "ElasticNet":{
                    'alpha':[0.1,0.5,1,2,5],
                    'l1_ratio':[0.1,0.25,0.5,0.75,1]
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)

            ## To get best model score from dict
            best_model_name, (best_model_score, best_model) = max(model_report.items(), key=lambda x: x[1][0])

            print("This is the best model and score:")
            print(best_model_name,"-", best_model_score)

            print("best params for the best model : ")
            print(best_model.get_params())

            ## best modelscore on test data
            print("Best model's r2 score on test data")
            r2 = r2_score(y_test,best_model.predict(X_test))
            print(r2)


            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            mlflow_logging(best_model, best_model.get_params(), X_test, y_test, best_model_name)

            r2_square = r2_score(y_test, predicted)
            return r2_square



        except Exception as e:
            raise CustomException(e,sys)