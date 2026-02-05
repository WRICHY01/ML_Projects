import logging
import uuid
from typing import Tuple
from typing_extensions import Annotated

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from zenml import step, get_step_context
from zenml.client import Client
from sklearn.base import RegressorMixin

from src.model_development import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

# active_stack = Client().active_stack
experiment_tracker = Client().active_stack.experiment_tracker
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
    Annotated[str, "zenml_run_id"],
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
                registered_model_name="customer_satisafaction_model"  #This might be an issue(confliction of names)
            )

            print("currently in the train_model_step")
            run_name = get_step_context().pipeline_run.id
            print(f"the object type of zenml_run_id  before typecasting is: {type(run_name)}")
            pipeline_name = get_step_context().pipeline.name
            print(f"pipline_name: {pipeline_name}")
            print(f"zenml_run_id: {run_name}")
            # print(f"get_step_context contents are: {get_step_context()}")
            # model_deployer: MLFlowModelDeploye/r = get_step_context().stack.model_deployer\
            # print(f"model_deployer: {model_deployer.__dir__()}")
            zenml_run_id = str(get_step_context().pipeline_run.id)
            # experiment_tracker.get_run_id(experiment_name="customer_satisfaction_deployment_experiment",
            # "customer_satisfaction_model_deployment",
            print(f"currently in the train_model func: {zenml_run_id} and has object type of {type(zenml_run_id)}")
      
            mlflow_client = MlflowClient()
            print(f'currently in the train model func trying to extrack\
                  mlflow_model_run id: {mlflow_client.__dir__()}')
            current_run = mlflow.active_run()
            if current_run  is None:
                raise RuntimeError(f"No active mlflow run found.")

            print(f"the object type of mlflow_run_id before typecasting is: {type(current_run.info.run_id)}")
            mlflow_run_id = current_run.info.run_id
            print(f"the mlflow_run_id is: {mlflow_run_id} and its new type after typecasting is: {type(mlflow_run_id)}")

            return trained_model, zenml_run_id, mlflow_run_id
        else:
            raise ValueError(f"Model name {config.model_name} not Supported")

    except Exception as e:
        logging.error(f"Error training the model: {e}")
        raise e