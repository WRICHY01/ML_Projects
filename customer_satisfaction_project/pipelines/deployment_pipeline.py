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
# from zenml.config import MlFlowDeploymentConfig
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from mlflow.tracking import MlflowClient, artifact_utils

from steps.ingesting_df_stp import ingest_df
from steps.cleaning_df_stp import clean_df
from steps.train_model_stp import train_model
from steps.evaluating_model_stp import evaluate_model
from steps.config import ModelNameConfig
# from steps.mlflow_model_deployer_step import custom_model_deployer
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
    # ['id', 'name', '_config', 'flavor', 'type', 'user', 
    # 'created', 'updated', 'labels', 'connector_requirements', 
    # 'connector', 'connector_resource_id', '_connector_instance', 
    # '__module__', '__annotations__', '__doc__', 'NAME', 'FLAVOR', 
    # '_service_path', 'config', 'get_service_path', 'local_path', 
    # 'get_model_server_info', 'perform_deploy_model', 
    # '_clean_up_existing_service', '_create_new_service', 
    # 'perform_stop_model', 'perform_start_model', 'perform_delete_model', 
    # '__abstractmethods__', '_abc_impl', 'get_active_model_deployer', 
    # 'deploy_model', 'find_model_server', 'stop_model_server', 
    # 'start_model_server', 'delete_model_server', 'get_model_server_logs', 
    # 'load_service', '__init__', 'from_model', 'settings_class', 'get_settings', 
    # 'connector_has_expired', 'get_connector', 'log_file', 'requirements', 
    # 'apt_packages', 'get_docker_builds', 'prepare_pipeline_deployment', 
    # 'get_pipeline_run_metadata', 'prepare_step_run', 'get_step_run_metadata', 
    # 'cleanup_step_run', 'post_registration_message', 'validator', 'cleanup', 
    # ]
    # Get the mlflow deployer stack component
    print(f"entered the prediction_service_loader and about to call the get_active_deployer func from mlflowdeployer")
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    print(f"\nThese are the contents of Mlflowdeployer component {mlflow_model_deployer_component.id},\
    {vars(mlflow_model_deployer_component)}\n")

    current_run = mlflow.active_run()
    # if current_run is None:
    #     raise RuntimeError(f"No active mlflow run found.")

    # print(f"the type of mlflow_run_id before typecasting is: {type(current_run.info.run_id)}")
    # mlflow_run_id = current_run.info.run_id
    # print(f"mlflow={mlflow_run_id}")
    print(f"get_step_context contents includes: {vars(get_step_context())}")
    print(f"running state is: {running}")
    # Fetch existing services with the same pipeline name, step_name and model_name
    existing_services = mlflow_model_deployer_component.find_model_server(
                        pipeline_name=pipeline_name,
                        pipeline_step_name=pipeline_step_name,
                        model_name=model_name,
                        # model_uri=mlflow_run_id,#This was included to dynamically select the right and recent deployed model.
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
# mlflow_deployer_finder = inspect.getsourcefile(Client().active_stack.model_deployer.deploy_model(config, ))
# if mlflow_deployer_finder:
#     print(f"\n\n The path for mlflow_deployer_finder is : {mlflow_deployer_finder}\n\n")
# else:
#     print("Couldnt find the file")
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
    # print(f"df_info: {df.in , ,
    # json_list = df.to_dict(orient="records")
    # print("data in json_list: ", json_list)
    # print(f"json_list value and data_type is as follows: {json_list}, {type(json_list)},\
    # {json_list[0]}, {type(json_list[0])}")
    # data = np.array(json_list)

    # invocations_url = f"{prediction_url}/invocations"

    # print("making prediction request to: ", invocations_url)
    # headers = {
    #     "Content-Type": "application/json"
    # }

    # formats_to_try = [
    #     {"dataframe_records": json_list},
    #     {"instances": json_list},
    #     {"inputs": json_list},
    #     json_list,
    #     {"columns": desired_columns, "data": [list(json_list[0].values())]}
    # ]

    # for i, format_data in enumerate(formats_to_try):
    #     try:
    #         print(f"\n--- Trying format {i+1} ---")
    #         print(f"Data format: {json.dumps(format_data, indent=2)[:300]}...")

    #         response = requests.post(
    #             invocations_url,
    #             json=format_data,
    #             headers=headers,
    #             timeout=30
    #         )

    #         print(f"Response status: {response.status_code}")
    #         print(f"Response text: {response.text[:200]}...")


    #         if response.status_code == 200:
    #             prediction = response.json()
    #             print(f"Prediction succesful with format {i+1}:{prediction}")

    #             if isinstance(prediction, list):
    #                 return np.array(prediction)

    #             else:
    #                 return np.array([prediction])

    #         else:
    #             print(f"Format {i+1} failed with status {response.status_code}")
    #     except Exception as format_error:
    #         print(f"Format {i+1} failed with exception: {format_error}")

    # try:

        
    #     print("\n--- Trying service.predict() method as fallback ---")
    #     data = {
    #         "dataframe_records": json_list
    #     }
    #     print(f"data prepared for prediction {data}")
    #     prediction_data = json.dumps(data)
    #     prediction = service.predict(prediction_data)
    #     print(f"predicition successfull via service.predict(): {prediction}")
    #     return prediction

    # except Exception as server_error:
    #     print(f"Service.predict() also failed: {server_error}")

    # raise RuntimeError("All prediction method failed. Check service logs and data format.")
    # data =  df.values
    # print("data in array format: ", data)
    # print(f"The final shape of data after transformation is: {data.shape}")
    

    
    # df = pd.DataFrame(data["data"], columns=desired_columns)
    # json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    # data = np.array(json_list)
    # prediction = service.predict(data)

    return prediction
@step
def deploy_model(run_id: str,
                _run_id: str,
                eval_run_id: str,
                _eval_run_id: str,
                model_name: str) -> str: #Optional[MLFlowDeploymentService]:

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
    # model_uri = artifact_utils.get_artifact_uri(
    #     run_id= mlflow_run_id,
    #     artifact_path=model_name
    # )
    # print("run_id if of type: ",type(run_id))
    # run_id = uuid.UUID(run_id)
    # print("run_id is now of type: ",type(run_id))
    # df7d7e150ae245cf8bf79caa23c6e75a
    # model_uri = f"runs:/{run_id}/{model_tag}"
    
    # return service
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

    # print(f"model_uri_id is: {model_uri_id}")

    # zenml_client = Client()
    # experiment_tracker = zenml_client.active_stack.experiment_tracker
    
    # run_name = get_step_context().pipeline_run.id
    # pipeline_name = get_step_context().pipeline.name
    # print(f"pipline_name: {pipeline_name}")
    # print(f"run_name: {run_name}")
    
    # mlflow_run_id_ = experiment_tracker.get_run_id(experiment_name="customer_satisfaction_deployment_experiment",
    # # "customer_satisfaction_model_deployment",
    #                                               run_name=run_name
    #                                              )
    # print(f"mlflow_run_id_ is : {mlflow_run_id_}")
    if deployment_decision:
        service = deploy_model(run_id=mlflow_run_id, _run_id=zenml_run_id,
        eval_run_id=eval_zenml_run_id, _eval_run_id=eval_mlflow_run_id,
        model_name="model")

    # mlflow_model_deployer_step(
    #     model=model,
    #     deploy_decision=deployment_decision,
    #     workers=workers,  
    #     timeout=timeout
    # )
# df = ingest_df(data_path)
# X_train, X_test, y_train, y_test = clean_df(df)
# model = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
# r2_, rmse_, zenml_run_id_, mlflow_run_id_ = evaluate_model(model, X_test, y_test)
# print(mlflow_run_id_)
@pipeline(enable_cache=False)
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    model_uri = load_model()
    data = dynamic_importer()
    # print(f"done calling the dynamic_importer func and about to call the prediction_service_loader")
    # service = prediction_service_loader(
    #     pipeline_name=pipeline_name,
    #     pipeline_step_name=pipeline_step_name,
    #     # running=True,
    # ) 
    print("done with prediction_service_loader")
    # service.start(timeout=20)
    
    prediction = predictor(model_uri=model_uri, data=data)
    # print(f"This is from the inference function: {service.__dir__()}")
    print("prediction:", vars(prediction))
    
    return prediction



