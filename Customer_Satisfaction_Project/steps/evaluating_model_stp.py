import logging
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
# from .train_model_stp import train_model
from src.model_evaluation import MSE, R2Score, RMSE


@step
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


        r2 = R2Score()
        r2score = r2.evaluate_model(y_test, y_pred)

        rmse = RMSE()
        rmse_score = rmse.evaluate_model(y_test, y_pred)

        return r2score, rmse_score
    except Exception as e:
        logging.error(f"Failed to Evaluate Model: {e}")
        raise e
