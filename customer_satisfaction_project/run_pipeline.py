import mlflow
from zenml.client import Client

from pipelines.training_pipeline import train_pipeline

uri = Client().active_stack.experiment_tracker.get_tracking_uri()

if __name__ == "__main__":
    # show experiment tracking url
    print(f"Experiment_tracking_url: {uri}")
    # Run Pipeline
    train_pipeline(data_path = r"\my_projects\customer_satisfaction_project\dataset\olist_customers_dataset.csv")


