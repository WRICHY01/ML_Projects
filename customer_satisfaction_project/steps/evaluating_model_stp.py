import logging
import uuid

import mlflow
from mlflow.tracking import MlflowClient
# from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step, get_step_context
from zenml.client import Client

from src.model_evaluation import MSE, R2Score, RMSE

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker="mlflow_tracker")
def evaluate_model(
    model: RegressorMixin, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame
    ) -> Tuple[
    Annotated[float, "r2score"],
    Annotated[float,"rmse_score"],
    Annotated[str, "eval_zenml_run_id"],
    Annotated[str, "eval_mlflow_run_id"]
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

        run_name = get_step_context().pipeline_run.id
        print(f"the type of zenml_run_id  before typecasting is: {type(run_name)}")
        pipeline_name = get_step_context().pipeline.name
        print(f"pipline_name: {pipeline_name}")
        print(f"zenml_run_id: {run_name}")
        # print(f"get_step_context contents are: {get_step_context()}")
        # model_deployer: MLFlowModelDeploye/r = get_step_context().stack.model_deployer\
        # print(f"model_deployer: {model_deployer.__dir__()}")
        eval_zenml_run_id = str(get_step_context().pipeline_run.id)
        # experiment_tracker.get_run_id(experiment_name="customer_satisfaction_deployment_experiment",
        # "customer_satisfaction_model_deployment",
        print(f"currently in the evaluate_model func: {eval_zenml_run_id} and has a type of {type(eval_zenml_run_id)}")
        
        
        mlflow_client = MlflowClient()
        current_run = mlflow.active_run()
        if current_run  is None:
            raise RuntimeError(f"No active mlflow run found.")

        print(f"the type of mlflow_run_id before typecasting is: {type(current_run.info.run_id)}")
        eval_mlflow_run_id = current_run.info.run_id
        print(f"the mlflow_run_id is: {eval_mlflow_run_id} and its new type after typecasting is: {type(eval_mlflow_run_id)}")

        return r2score, rmse_score, eval_zenml_run_id, eval_mlflow_run_id
    except Exception as e:
        logging.error(f"Failed to Evaluate Model: {e}")
        raise e


# def call_ems_return_values():
