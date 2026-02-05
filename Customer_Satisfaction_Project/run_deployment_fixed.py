import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
    )
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from typing import cast
import os

from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline

# Fix the data path - use absolute path or relative from project root
data_path = os.path.join(os.getcwd(), "dataset", "olist_customers_dataset.csv")

DEPLOY = "deploy"
PREDICT = "predict"  # Fixed: was "Predict" (capital P)
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
def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT  # Fixed: was checking DEPLOY twice
    
    if deploy:
        continuous_deployment_pipeline(
            data_path,
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60
        )
    
    if predict:
        inference_pipeline()

    # Get the tracking URI
    tracking_uri = get_tracking_uri()
    
    # Fixed: Corrected the MLFlow UI command and markup
    print(
        "\n🚀 [bold green]Deployment Complete![/bold green]\n\n"
        "📊 [bold blue]View your experiments in MLFlow UI:[/bold blue]\n"
        f"[italic green]mlflow ui --backend-store-uri {tracking_uri}[/italic green]\n\n"
        "🌐 Then open: [link]http://localhost:5000[/link]\n\n"
        "📈 [bold]What you can do in MLFlow UI:[/bold]\n"
        "   • Compare model performances across runs\n"
        "   • View training metrics and parameters\n"
        "   • Download model artifacts\n"
        "   • Manage model versions\n"
        "   • Track experiment lineage\n\n"
    )

    # Alternative: Start MLFlow UI automatically (Windows-friendly)
    print(
        "💡 [bold yellow]Windows Users:[/bold yellow] Since daemon functionality isn't supported,\n"
        "   you can start MLFlow UI manually with the command above.\n\n"
    )

    # Fetch existing services with same pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"
    )
    
    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"🟢 [bold green]MLFlow prediction server is running![/bold green]\n"
                f"📡 Prediction URL: [link]{service.prediction_url}[/link]\n"
                f"🛑 To stop the service run:\n"
                f"[italic green]zenml model-deployer models delete {str(service.uuid)}[/italic green]\n"
            )
        elif service.is_failed:
            print(
                f"🔴 [bold red]MLFlow prediction server failed:[/bold red]\n"
                f"📊 Last state: '{service.status.state.value}'\n"
                f"❌ Last error: '{service.status.last_error}'\n"
            )
    else:
        print(
            "⚠️  [bold yellow]No MLFlow prediction server is currently running.[/bold yellow]\n"
            "🚀 The deployment pipeline must run first to train a model and deploy it.\n"
            f"💡 Execute: [italic green]python run_deployment.py --config {DEPLOY}[/italic green]\n"
        )

    # Provide additional helpful commands
    print(
        "\n🔧 [bold blue]Useful Commands:[/bold blue]\n"
        f"• View experiments: [italic green]mlflow ui --backend-store-uri {tracking_uri}[/italic green]\n"
        "• List ZenML services: [italic green]zenml model-deployer models list[/italic green]\n"
        "• Check ZenML stack: [italic green]zenml stack describe[/italic green]\n"
    )

if __name__ == "__main__":
    run_deployment()