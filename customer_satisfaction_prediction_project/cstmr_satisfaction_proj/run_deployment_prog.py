import os
import click
from rich import print
from typing import cast


import json
import pandas as pd
import mlflow
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer,)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


from cstmr_satisfaction_proj.pipelines.deployment_pipeline_wsl import deploy_model, continuous_deployment_pipeline, inference_pipeline


script_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_path, "dataset", "olist_customers_dataset.csv")
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# @step
# def get_step_info():# = get_step_context()
#     get_step_context()

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model(`deploy`), or to "
    "only run a prediction against the deployed model (`predict`)."
    "By default both will be run (`deploy_and_predict`)"

)
@click.option(
    "--min_accuracy",
    default=0.0,
    help="Minimum accuracy required to deploy the model."
)
@click.option(
    "--data_path",
    default=str(file_path),
    type=click.Path(exists=True)
)
@click.option(
    "--workers",
    default=3,
    type=int,
    help="No of workers for Pipeline execution."
)
@click.option(
    "--timeout",
    default=DEFAULT_SERVICE_START_STOP_TIMEOUT,
    type=int,
    help="Timeout in secs for pipeline execution."
)
def run_deployment(config: str, min_accuracy: float, 
                    data_path: str, workers: int, timeout: int):
    
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config in [DEPLOY, DEPLOY_AND_PREDICT]
    predict = config in [PREDICT, DEPLOY_AND_PREDICT]
    if deploy:
        deploy_mlflow_settings = MLFlowExperimentTrackerSettings(
                    experiment_name="customer_satisfaction_deployment_experiment",
                    nested=True,
                    tags={"project": "customer_satisfaction_deployment"})
        
        configured_continuous_deployment_pipeline = continuous_deployment_pipeline.with_options(
            settings={"docker":docker_settings,
                      "experiment_tracker": deploy_mlflow_settings}
            
        )
        configured_continuous_deployment_pipeline(
            data_path,
            min_accuracy=min_accuracy,
            workers=workers,
            timeout=timeout)

    if predict:
        print(f"timeout is {timeout/60}")
        inference_mlflow_settings = MLFlowExperimentTrackerSettings(
                    experiment_name="customer_satisfaction_inference_experiment",
                    nested=True,
                    tags={"project": "customer_satisfaction_inference"})
        
        configured_inference_pipeline = inference_pipeline.with_options(
            settings={"docker":docker_settings,
                      "experiment_tracker": inference_mlflow_settings
            }
        )
        configured_inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            # pipeline_name="mflow_model_deployer_step",
            pipeline_step_name="deploy_model"
        )

    print(
        "You can run:\n "
        f"[italic green] mlflow ui-backend-store-ui '{get_tracking_uri()}'"
        "[/italic green]\n... to inspect your experiment runs within the MLflow"
        "UI.\n You can find your runs tracked within the "
        "`mlflow_example_pipeline` experiment. There you'll also be able to  "
        "compare two or more runs.\n\n"
    )

    # Fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="deploy_model",
        model_name="model"
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        print(f"service info entails: {service}")
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as daemon "
                f"process service and accepts inference request at:\n"
                f" {service.prediction_url}\n"
                f"To stop the service run"
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            f"No MLflow prediction server is currently running. The deployment "
            f"pipeline must run first to train a model and deploy it. Execute "
            f"the same command with the `--deploy` argument to deploy a model."
        )

if __name__ == "__main__":
    run_deployment()