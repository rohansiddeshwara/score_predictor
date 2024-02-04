from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import os

if __name__ == "__main__":
    try:
        logging.info("Hello World")
        print(f"Hello World from score_predictor")
        save_object("artifacts/sample.pkl", "Hello World")
        logging.info("successfully saved the object")
    except CustomException as e:
        logging.error(e)
