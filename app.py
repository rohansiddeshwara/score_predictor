from flask import Flask
from flask
from flask_cors import CORS


from src.pipelines.training_pipeline import TraningPipeline 


from src.logger import logging
from src.exception import CustomException


app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    try:
        logging.info("Hello World")
        return "Hello, World!"
    except CustomException as e:
        logging.error(e)
        return "Error occured"
    

@app.route("/train")
def train():
    try:
        logging.info("Training Started")
        traning_pipeline = TraningPipeline()
        traning_pipeline.initiate_traning()
        return "Training Started"
    except CustomException as e:
        logging.error(e)
        return "Error occured"

if __name__ == "__main__":
    logging.info("Starting the app")
    app.run(host="0.0.0.0", port=5000)
