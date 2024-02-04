import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object


@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join("artifacts", 'raw_data.csv')
    test_data_path: str = os.path.join("artifacts", 'test_data.csv')
    train_data_path: str = os.path.join("artifacts", 'train_data.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")
            logging.info("Readign from csv file")
            data = pd.read_csv(self.ingestion_config.data_path) 

            target_coulumn = "target"

            logging.info("Splitting data into train and test")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            logging.info("Saving train and test data")
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            

            logging.info("Data Ingestion Completed")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            logging.error(f"Data Ingestion Failed: {e}")
            raise CustomException(f"Data Ingestion Failed: {e}",sys)
        