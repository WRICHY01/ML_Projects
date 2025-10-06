import logging
import uuid
from typing import Tuple
from typing_extensions import Annotated

import mlflow
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin

from src.model_development import LinearRegressionModel
from .config import ModelNameConfig

# active_stack = Client().active_stack
# experiment_tracker = Client().active_stack.experiment_tracker
# print(experiment_tracker)
@step(experiment_tracker="mlflow_tracker")
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig
) -> Tuple[
    Annotated[RegressorMixin, "trained_model"],
    Annotated[str, "mlflow_run_id"]
]:
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
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)

            mlflow.sklearn.log_model(
                sk_model=trained_model,
                artifact_path="model",
                registered_model_name="customer_satisafaction_model"
            )

            current_run = mlflow.active_run()
            if current_run  is None:
                raise RuntimeError(f"No active mlflow run found.")

            mlflow_run_id = current_run.info.run_id

            return trained_model, mlflow_run_id
        else:
            raise ValueError(f"Model name {config.model_name} not Supported")

    except Exception as e:
        logging.error(f"Error training the model: {e}")
        raise e