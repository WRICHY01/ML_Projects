import logging

from zenml import pipeline
import pandas as pd

from steps.ingesting_df_stp import ingest_df
from steps.cleaning_df_stp import clean_df
from steps.train_model_stp import train_model
from steps.evaluating_model_stp import evaluate_model
from steps.config import ModelNameConfig

@pipeline(enable_cache = True)
def train_pipeline(data_path: str):
    """
    This creates a pipeline for data ingestion, data cleaning, train model, and evaluate model
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    r2_score, rmse_score = evaluate_model(model, X_test, y_test)


# train_pipeline()
