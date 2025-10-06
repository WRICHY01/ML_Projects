import os
import logging
import uuid
from pydantic import BaseModel
import json
import inspect
import time
import requests
from typing import Optional

import numpy as np
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from zenml import step, pipeline, get_step_context
from zenml.client import Client
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer,)
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentConfig
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from mlflow.tracking import MlflowClient, artifact_utils

from steps.ingesting_df_stp import ingest_df
from steps.cleaning_df_stp import clean_df
from steps.train_model_stp import train_model
from steps.evaluating_model_stp import evaluate_model
from steps.config import ModelNameConfig
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseModel):
    """
    Deployment Trigger Config
    """
    min_accuracy: float = 0.0

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
) -> bool:
    """
    Implements a simple deployment triggering mechanism when the condition is 
    met(deploys the model when the condition is met).
    i.e when the model metric is greater than the minimum_accuracy(min_accuracy)
    """
    return accuracy >= config.min_accuracy 

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str="model",
    # model_uri: str, 
    running=True
    ) -> MLFlowDeploymentService:
    """
    Get the prediction service started by the deployment pipeline
    Args:
        pipeline_name: The name of the pipeline deployed by the MLFlow prediction server

    """
    # Get the mlflow deployer stack component
    print(f"entered the prediction_service_loader and about to call the get_active_deployer func from mlflowdeployer")
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    current_run = mlflow.active_run()
    # if current_run is None:
    #     raise RuntimeError(f"No active mlflow run found.")

    print(f"get_step_context contents includes: {vars(get_step_context())}")
    print(f"running state is: {running}")
    # Fetch existing services with the same pipeline name, step_name and model_name
    existing_services = mlflow_model_deployer_component.find_model_server(
                        pipeline_name=pipeline_name,
                        pipeline_step_name=pipeline_step_name,
                        model_name=model_name,
                        running = running
                        )
    print(f"The no of services found is: {len(existing_services)}")
    print(existing_services)
    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment service found for pipeline `{pipeline_name}`,"
            f"step `{pipeline_step_name}` and model `{model_name}`"
        )

    return existing_services[0]

@step
def predictor(
    model_uri: str,
    data: str
) -> np.ndarray:
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    desired_columns = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    df = pd.DataFrame(data["data"])
    df = df.T
    df.columns = desired_columns
    prediction = model.predict(df)
    print(f"predictions: {prediction}")
    

    return prediction
@step
def deploy_model(run_id: str,
                _run_id: str,
                eval_run_id: str,
                _eval_run_id: str,
                model_name: str) -> str:

    print("currently in the deploy_model func and it current experiment run_id is: ", run_id)
    deployment_info = {
        "run_id": run_id,
        "model_name": model_name,
        "model_uri": f"runs:/{run_id}/{model_name}",
        "deployment_time": time.time()
    }

    with open("model_deployment.json", "w") as f:
        json.dump(deployment_info, f)

    print(f"Model deployment info saved for run_id: {run_id}")
    return f"runs:/{run_id}/{model_name}"
    
@step
def load_model() -> str:
    """
    Load the deployed model directly without a server
    """
    if not os.path.exists("model_deployment.json"):
        raise RuntimeError("No model deployement info found")

    with open("model_deployment.json", "r") as f:
        deployment_info = json.load(f)

    return deployment_info["model_uri"]

@pipeline(enable_cache=False)
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.0,
    workers: int = 1,
    timeout: int = 300
) -> None:
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    # The mlflow_run_id was returned from the train_model step for deployment purpose,
    # This is done to enable location of the model when deploying due to that is where the saved model by mlflow is located.
    model, zenml_run_id, mlflow_run_id = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    r2_score, rmse_score, eval_zenml_run_id, eval_mlflow_run_id = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(r2_score, DeploymentTriggerConfig())

    if deployment_decision:
        service = deploy_model(run_id=mlflow_run_id, _run_id=zenml_run_id,
        eval_run_id=eval_zenml_run_id, _eval_run_id=eval_mlflow_run_id,
        model_name="model")


@pipeline(enable_cache=False)
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    model_uri = load_model()
    data = dynamic_importer()
    prediction = predictor(model_uri=model_uri, data=data)
    
    return prediction



