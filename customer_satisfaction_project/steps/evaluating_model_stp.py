import logging

import mlflow
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from src.model_evaluation import MSE, R2Score, RMSE

experiment_tracker = Client().active_stack.experiment_tracker
# print(experiment_tracker.experiment_name)

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame
    ) -> Tuple[
    Annotated[float, "r2score"],
    Annotated[float,"rmse_score"]
]:
    """
    Evaluate the model using X metrics

    Args:
        model(): The trained model

    Returns:
        
    """
    try:
        y_pred = model.predict(X_test)
        mse = MSE()
        mse_score = mse.evaluate_model(y_test, y_pred)
        mlflow.log_metric("mse_score", mse_score)


        r2 = R2Score()
        r2score = r2.evaluate_model(y_test, y_pred)
        mlflow.log_metric("r2score", r2score)

        rmse = RMSE()
        rmse_score = rmse.evaluate_model(y_test, y_pred)
        mlflow.log_metric("rmse_score", rmse_score)

        return r2score, rmse_score
    except Exception as e:
        logging.error(f"Failed to Evaluate Model: {e}")
        raise e
