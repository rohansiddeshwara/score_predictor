import os
import sys
from urllib.parse import urlparse
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from src.exception import CustomException
from src.logger import logging



mlflow.set_registry_uri("https://dagshub.com/rohansiddeshwara/score_predictor.mlflow")
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def mlflow_logging(best_model, best_params, X_test, y_test, actual_model):

    with mlflow.start_run():

        predicted_qualities = best_model.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        mlflow.log_params(best_params)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)


        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
        else:
            mlflow.sklearn.log_model(best_model, "model")

    logging.info(f"Best found model on both training and testing dataset")
