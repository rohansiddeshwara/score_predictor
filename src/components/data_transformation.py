import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    processor_object_path: str = os.path.join("artifacts", 'processor_object.pkl')


class DataTransformation:
    """
    This function is used to create data treansformation object
    
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_processor(self, raw_data, target_column): 
        try:
            logging.info("Data Transformation Started")
            logging.info("Data Preprocessing Started")

            data  = pd.read_csv(raw_data)
            y = data.drop(columns=[target_column],axis = 1, inplace=True)

            numeric_features = data.select_dtypes(include=["number"]).columns.tolist()
            categorical_features = data.select_dtypes(include=["object_"]).columns.tolist()


            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)])

            logging.info("Data Preprocessing Completed")
            logging.info("Data Transformation Completed")

            return preprocessor
        except Exception as e:
            logging.error(f"Data Transformation Failed: {e}")
            raise CustomException(f"Data Transformation Failed: {e}",sys)
        

    def initiate_data_transformation(self,raw_data_path, train_data_path,test_data_path, target_column):
        try:

            logging.info("Reading train and test data")
            test_data = pd.read_csv(test_data_path)
            train_data = pd.read_csv(train_data_path)

            logging.info("removing target column from data")
            X_train = train_data.drop(target_column, axis=1)
            y_train = train_data[target_column]

            X_test = test_data.drop(target_column, axis=1)
            y_test = test_data[target_column]

            logging.info("applying preprocessor on train and test data")
            processor = self.get_data_processor(raw_data_path, target_column)
            logging.info("recived processor object")

            X_train = processor.fit_transform(X_train)
            X_test = processor.transform(X_test)


            train_array = np.c_[X_train, y_train]
            test_array = np.c_[X_test, y_test]

            logging.info("saving th processor object")
            save_object(obj=processor,file_path=self.transformation_config.processor_object_path)
            

            return (train_array, test_array )
        except Exception as e:
            CustomException(f"Data Ingestion Failed: {e}",sys)

