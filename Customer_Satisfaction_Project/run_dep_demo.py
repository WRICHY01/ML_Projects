import click
from rich import print
from typing import cast

from zenml.integration.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integration.mlflow.model_services import MLFlowDeploymentService
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline

data_path = r"\my_projects\customer_satisfaction_project\dataset\olist_customers_dataset.csv"

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model(`deploy`), or to only "
    "run a prediction against the deployed model(`predict`)."
    "By default both will run (`deploy_and_predict`)"
)
@click.option(
    "--min_accuracy",
    default=0.0,
    type=float,
    help="Minimum accuracy to Deploy the model."
)
@click.option(
    "--data_path",
    default=str(data_path)
    type=click.Path(exits=True)
)
@click.option(
    "--workers",
    default=3,
    type=int,
    help="No of workers for Pipeline execution."
)
@click.option(
    "--timeout",
    default=60,
    type=int,
    help="Timeout in secs for pipeline execution."
)
def run_deployment(config:str, min_accuracy:float, data_path: str, workers: int, timeout: int):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config in [DEPLOY, DEPLOY_AND_PREDICT]
    predict = config in [PREDICT, DEPLOY_AND_PREDICT]

    print(f"[bold_blue]Starting MLOps workflow with config: {config}[/bold_blue]")

    if deploy:
        try:
            print(f"[bold_green] Running Deployment Pipeline...[/bold_green]")
            continuous_deployment_pipeline(
                data_path,
                min_accuracy=min_accuracy,
                workers=workers,
                timeout=timeout
            )
        except Exception as e:
            print(f"[bold_red] Deploying Pipeline Failed: {e}[/bold_red]")
            # if not predict:
            #     sys.exit(1)

    elif predict:
        try:
            print(f"[bold_green] Running inference pipeline..[/bold_green]")
            inference_pipeline(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step"
            )

            print(f"[bold_green] Inference pipeline completed successfully..[/bold_green]")
        except Exception as e:
            print(f"[bold_red] Failed to make Inference: {e}[/bold_red]")

    _display_mlflow_info()

    _check_service_status()

def _display_mlflow_info():
    """
    Displaying MLFlow UI access informatio.
    """
    try:
        tracking_uri = get_tracking_uri()
        print(f"[bold cyan] MLFlow Tracking [/bold cyan]"
              f"You can run:\n"
              f"[italic green] mlflow ui --backend-store-uri '{tracking_uri}'[/italic green]"
              f"...To inspect your experiment runs within the MLFlow UI. \n"
              f"Find your runs in the `mlflow_example_pipeline` experiment."
              f"where you can compare multiple runs.\n [/bold cyan]"
              )
    except Exception as e:
        print(f"[yellow] Could not retrieve the MLFlow Tracking URI: {e}[/yellow]")

def _check_service_status():
    """
    Check and Display the status of deployed model services.
    """
    try:
        existing_services = mlflow_model_deployer_component.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step"
            model_name="model"
        )

        if existing_services:
            service = cast(MLFlowDeploymentService, existing_services[0])
            if service.is_running:
                print(f"[bold green] Model Service Status: Running[/bold green]"
                      f"The MLFlow prediction server is running as a daemon process."
                      f"[bold] Prediction URL:[/bold] {service.prediction_url}"
                      f"\n[italic] To stop the service:[/italic]"
                      f"[italic green] zenml model-deployer models delete {service.uuid}.[/italic green]")

            elif service.is_failed:
                print(f"[bold red] Model service status: FAILED [/bold_red]"
                    f"[bold] last state: {service.status.state.value}"
                    f"[bold] last error: {service.status.last_error}")

            else:
                print(f"\n[bold yellow] Model service status: {service.status.state.value.upper()}[/bold yellow]")

        else:
            print(f"[bold yellow] No model Service found [/bold yellow]"
                  f"No MLFlow prediction server is currently running.."
                  f"The deployment Pipeline must run first to train and deploy a model"
                  f"Execute: [italic green] python {__file__} --config deploy [italic green]")

    except Exception as e:
        print(f"[bold yellow] Could not check service status: {e} [/bold yellow]")



if __name__ == "__main__":
    run_deployment()