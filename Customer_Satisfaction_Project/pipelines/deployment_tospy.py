import logging
from pydantic import BaseModel
import json

import numpy as np
import pandas as pd
import zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow_services import MLFlowDeploymentService

from steps.ingesting_df_stp import ingest_df
from steps.cleaning_df_stp import clean_df
from steps.train_model_stp import train_model
from steps.evaluating_model_stp import evaluate_model
from steps.config import ModelNameConfig
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    """ Deployment Trigger Config """
    min_accuracy: float = 0.0

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()

    return data

@step
def deployment_trigger(
    accuracy: float, 
    config: DeploymentTriggerConfig) -> bool:
    """
    Implements a simple deployment triggering mechanism when the condition is 
    met(deploys the model when the condition is met).
    i.e when the model metric is greater than the min_accuracy(minimum accuracy)
    """
    accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """
    MLFlow deployment Getter Parameters.

    Attributes:
        pipeline_name: The name of the pipeline that deployed the MLFlow Prediction Server
        step_name: The name of the step that deployed the MLFlow prediction server
        running: The step only returns a running service when this flag is set.
        model_name: The name of the model that was deployed
    """
    pipeline_name: str
    step_name: str
    running: bool = True
    model_name: str

@step(enable_cache=False)
def  prediction_service_loader(
    params: MLFlowDeploymentLoaderStepParameters,
) -> MLFlowDeploymentService:
    """
    Get the prediction service started by the deployment pipline
    """
    # Get the mlflow deployen stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    # Fetch existing services with the same pipeline name, and model_name
    existing_services = mlflow_model_deployer_component.find_model_server(
                            pipeline=params.pipeline_name,
                            step_name=params.step_name,
                            running=params.running
                            model_name=params.model_name
                        )
    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment service found for pipeline `{params.pipeline_name}`"
            f"step `{params.step_name}` and model `{params.model_name}`"
        )
    
    return existing_services[0]

@pipeline
def predictor(
    service: MLFlowDeploymentService,
    data: np.array
) -> np.array:
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    desired_columns = [
        "payment_sequential",
        "payment_installments",
        "payment_values",
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

    df = pd.DataFrame(data["data"], columns=desired_columns)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)

    return prediction


# Continuous Deployment Pipeline 
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deploymen_pipeline(
    data_path: str,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    r2_score, rmse_score = evaluate_model(model)
    deployment_decision = deployment_trigger(r2_score, DeploymentTriggerConfig())
    mlflow_model_deployer_step(
        model=model,
        deploy_decison=deployment_decision,
        workers=workers,
        timeout=timeout
    )

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def inference_pipeline(
    params: MLFlowDeploymentLoaderStepParameters):
    data = dynamic_importer()
    service = prediction_service_loader(params)