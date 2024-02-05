import pandas as pd
import numpy as np 
import os
import sys

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_tranier import ModelTrainer

class TraningPipeline:
    def __init__(self):
        self.data_ingestion  = DataIngestion()
        self.data_transformation = DataTransformation()
        self.trainer = ModelTrainer()
        self.target_column = "math_score"
        self.raw_data_path = os.path.join("artifacts", 'raw_data.csv')
        
    def initiate_traning(self):
        try:
            logging.info("Data Preparation Started")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            train_array,test_array = self.data_transformation.initiate_data_transformation(self.raw_data_path, train_data_path,test_data_path, self.target_column)

            logging.info("Data Preparation Completed")

            logging.info("Model Training Started")
            r2_score = self.trainer.initiate_model_trainer(train_array,test_array)
            logging.info("Model Training Completed")

            logging.info(f"R2 Score: {r2_score}")

        except Exception as e:
            logging.error(f"Data Preparation Failed: {e}")
            raise CustomException(f"Data Preparation Failed: {e}",sys)
        



if __name__ == "__main__":
    traning_pipeline = TraningPipeline()
    traning_pipeline.initiate_traning()