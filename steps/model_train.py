import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains the model on the Ingested Data

    Args: 
        X_train: Training Data
        X_test: Test Data
        y_train: Train Labels
        y_test: Test Labels
    """
    try:
        model = None
        if config.model_name == "Linear Regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported!")
    except Exception as e:
        logging.error(f"Error in Training Model : {e}")
        raise e
    