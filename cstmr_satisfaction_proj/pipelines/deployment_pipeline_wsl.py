import logging
from pydantic import BaseModel
import json
import requests
from typing import Optional
import inspect

import numpy as np
import pandas as pd
from zenml import step, get_step_context, pipeline
from zenml.client import Client
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer,)
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentConfig
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from steps.ingesting_df_stp import ingest_df
from steps.cleaning_df_stp import clean_df
from steps.train_model_stp import train_model
from steps.evaluating_model_stp import evaluate_model
from steps.config import ModelNameConfig
from .utils import get_data_for_test

class DeploymentTriggerConfig(BaseModel):
    """
    Deployment Trigger Config
    """
    min_accuracy: float = 0.0

@step
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
) -> bool:
    """
    Implements a simple deployment triggering mechanism
    """
    return accuracy >= config.min_accuracy

@step
def deploy_model(
    run_id: str,
    model_name: str,
    running: bool=False
) -> Optional[MLFlowDeploymentService]:
    
    zc = Client()
    model_deployer = zc.active_stack.model_deployer
    experiment_tracker = zc.active_stack.experiment_tracker
    # mlflow_model_component = MLFlowModelDeployer.get_active_model_deployer()
        
    step_context = get_step_context()
    print(f"we are here again get_step_context is thus: {step_context}")
    pipeline_name = step_context.pipeline.name
    pipeline_step_name = "deploy_model"
    # existing_services = mlflow_model_component.find_model_server()
    existing_services = model_deployer.find_model_server(
                        pipeline_name=pipeline_name,
                        pipeline_step_name=pipeline_step_name,
                        model_name=model_name,
                        running=running
    )

    if not existing_services:
        raise RuntimeError("No model Found")
    print(f"The No of service/s is/are: {len(existing_services)}")

    model_uri = f"runs:/{run_id}/{model_name}"

    mlflow_deployment_config = MLFlowDeploymentConfig(
        name="customer_satisfaction_model_deployment",
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="deploy_model",
        model_uri=model_uri,
        model_name=model_name,
        workers=1,
        mlserver=False, 
        blocking=False,
        silent_daemon=False,
        timeout = 900
    )
    
    service = model_deployer.deploy_model(
        config=mlflow_deployment_config,
        service_type=MLFlowDeploymentService.SERVICE_TYPE
    )

    

@step
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str="model",
    running: bool=False
) -> MLFlowDeploymentService:
    """
    Get the prediction service started by the deployment pipeline
    Args:
        pipeline_name: The name of the pipeline deployed by the model deployer
    """
    mlflow_deployment_component = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = mlflow_deployment_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
        )

    if not existing_services:
        raise RuntimeError(
            f"No Mlflow deployment service found for pipeline `{pipeline_name}`,"
            f" step `{pipeline_step_name}` and model `{model_name}`"
        )

    return existing_services[0]

@step
def predictor(
    data: str,
    service: MLFlowDeploymentService
) -> np.ndarray:
    service.start(timeout=600)
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
    json_data = df.to_dict(orient="split")
    payload = json.dumps(
        {
            "dataframe_split": json_data
            }
    )

    # you'd have to make some tweaks in the MLFlowDeploymentService source code in order 
    # to run inference using the `service.predict(arg), by changing the API schema from
    # "instances" to "dataframe_split" in the predict method
    
    # predictions = service.predict(df)
    # if predictions.shape[0] > 0:
    #     print(type(predictions))
    #     for i, prediction in enumerate(predictions):
    #         print(f"dataset {i + 1}. => {prediction}")
    
    # return predictions
    response = requests.post(
        url=service.get_prediction_url(),
        json={"dataframe_split": json_data},
        # headers={"Content-Type":"application/json"}
    )

    response.raise_for_status()
    predictions = np.array(response.json()["predictions"])
    # print(np.array(response.json()["predictions"]))
    if predictions.shape[0] > 0:
        for i, prediction in enumerate(predictions):
            print(f"dataset {i + 1}. => {int(prediction)}")
    return predictions

@pipeline(enable_cache=False)
def continuous_deployment_pipeline(
                        data_path: str,
                        min_accuracy: float,
                        workers: int,
                        timeout: int
                          ) -> None:
    """
    The mlflow_run_id was returned from the train_step for deployment purpose,
    This was done to enable location of the model when deploying due to that is where
    run_id of the saved model created by mlflow is located
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model, mlflow_run_id  = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    r2score, rmse_score = evaluate_model(model, X_test, y_test)
    deployment_condition = deployment_trigger(r2score, DeploymentTriggerConfig())

    #Testing the waters here, mlflow_model_deployer_step or deploy_model() step function can be used
    # They practically do the same thing: `Deploy your model`, deploy_model() step function is just 
    # more flexible and give more control as compared to the former.
    # mlflow_model_deployer_step(
    #     model=model,
    #     deploy_decision=deployment_condition,
    #     workers=workers,
    #     timeout=timeout,
    # )
    if deployment_condition:
        deployment_service = deploy_model(
                                        run_id=mlflow_run_id,
                                        model_name="model"
                                          )

@pipeline(enable_cache=0)
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
) -> None:
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name
    )
    prediction = predictor(data, service)