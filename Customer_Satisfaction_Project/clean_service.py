import click
from rich import print
from typing import cast

from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer,)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from typing import cast

from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline

data_path = r"\my_projects\customer_satisfaction_project\dataset\olist_customers_dataset.csv"


YES = "yes"
NO = "no"

@click.command()
@click.option(
    "--action",
    default=NO,
    help="action to clean already deployed service"
)

def clean_failed_service(action: str):
    zenml_client = Client()
    # print("vars of zenml_client", vars(zenml_client), zenml_client.__dir__())
    deployments = zenml_client.list_deployments()
    model_deployer = zenml_client.active_stack.model_deployer
    svcs = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step"
    )

    # print("vars of svc:", svcs.__dir__())
    print("found ", len(svcs))
    # print(vars(svcs))
    # svc = svcs.pop(svcs.uuid)
    # print("*" * 300)
    # for i, deployment in enumerate(deployments):
    #     print("deleting deployment: ", i, deployment.id, deployment)
    #     zenml_client.delete_deployment(deployment.id)
    print("model_lists is/are: ", deployments)
    # print("successfully deleted all deployments..")
    # print("*" * 300)

    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    print(f"\n The mlflow_model_deployer_component says: {mlflow_model_deployer_component}\n")
    try:
        if YES:
            services = mlflow_model_deployer_component.find_model_server(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                model_name="model"
            )

            print(f"found {len(services)} services.")
            if not services:
                print("No service found.")
            # if services.failed:
            print("vars of mlflow_model_deployer_component:", vars(mlflow_model_deployer_component))#, mlflow_model_deployer_component._dir__())
            for service in services:
                try:
                    print("vars:", vars(service))
                    print("dir:", service.__dir__())
                    service.stop(timeout=10)
                    print("service stopped...")
                except Exception as e:
                    print(f"Failed to stop service: {e}")
    except Exception as e:
        print(f"clean_up failed: {e}")

    #     continuous_deployment_pipeline(
    #         data_path,
    #         min_accuracy=min_accuracy,
    #         workers=3)
    # if predict:
    #     inference_pipeline(
    #         pipeline_name="continuous_deployment_pipeline",
    #         pipeline_step_name="mlflow_model_deployer_step"
    #     )

    # print(
    #     "You can run:\n "
    #     f"[italic green] mlflow ui-backend-store-ui '{get_tracking_uri}'"
    #     "[/italic green]\n... to inspect your experiment runs within the MLflow"
    #     "UI.\n You can find your runs tracked within the "
    #     "`mlflow_example_pipeline` experiment. There you'll also be able to  "
    #     "compare two or more runs.\n\n"
    # )

    # # Fetch existing services with same pipeline name, step name and model name
    # existing_services = mlflow_model_deployer_component.find_model_server(
    #     pipeline_name="continuous_deployment_pipeline",
    #     pipeline_step_name="deploy_model",
    #     model_name="model"
    # )
    # print(f"The existing_services says: {existing_services}")

    # if existing_services:
    #     service = cast(MLFlowDeploymentService, existing_services[0])
    #     print(f"\n\n currently in the run_deployment script {vars(service)}\n\n")
    #     if service.is_running:
    #         print(
    #             f"The MLflow prediction server is running locally as daemon "
    #             f"process service and accepts inference request at:\n"
    #             f" {service.prediction_url}\n"
    #             f"To stop the service run"
    #             f"[italic green]`zenml model-deployer models delete "
    #             f"{str(service.uuid)}`[/italic green]."
    #         )
    #     elif service.is_failed:
    #         print(
    #             f"The MLflow prediction server is in a failed state:\n"
    #             f" Last state: '{service.status.state.value}'\n"
    #             f" Last error: '{service.status.last_error}'"
    #         )
    # else:
    #     print(
    #         f"No MLflow prediction server is currently running. The deployment "
    #         f"pipeline must run first to train a model and deploy it. Execute "
    #         f"the same command with the `--deploy` argument to deploy a model."
    #     )

if __name__ == "__main__":
    clean_failed_service()