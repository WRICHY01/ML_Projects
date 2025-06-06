import logging

import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin

from src.model_development import LinearRegressionModel
from .config import ModelNameConfig

# active_stack = Client().active_stack
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
    Trains the model on the ingested data.

    Args 
        X_train: Training data
        X_test: Testing data
        y_train: Training Labels
        y_test: Testing Labels

    Returns:

    """
    try:
        # model = None
        if config.model_name == "LinearRegression":
            # mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model name {config.model_name} not Supported")

    except Exception as e:
        logging.error(f"Error training the model: {e}")
        raise e