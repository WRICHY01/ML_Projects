import os

import mlflow
from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

uri = Client().active_stack.experiment_tracker.get_tracking_uri()

script_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_path, "dataset", "olist_customers_dataset.csv")

if __name__ == "__main__":
    # show experiment tracking url
    print(f"Experiment_tracking_url: {uri}")
    # Run Pipeline
    train_pipeline(data_path = file_path)


