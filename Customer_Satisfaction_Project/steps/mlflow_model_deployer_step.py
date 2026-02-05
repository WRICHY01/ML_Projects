from typing import Any

from zenml import step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (MLFlowModelDeployer,)
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentConfig

@step
def custom_model_deployer(model: Any, 
                          deployment_decision: bool, 
                          model_name: str, 
                          timeout)->None:
    if not deployment_decision:
        print("Skipping deployment: Model didnt meet the threshold. Exiting...")
        return
    
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    deployment = deployer.deploy_model(
        config=MLFlowDeploymentConfig(
            model_name=model_name,
            model_uri=model.uri,
            service_name="zenml-model",
            blocking=True,
            silent=True,
            timeout=timeout
        ),
        model=model
    )

