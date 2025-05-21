import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreprocessStrategy, DataSplitStrategy, DataCleaningStrategy
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:

    """
    Cleans the Dataset and Splits it into training and Test Dataset

    Args;
        pd.DataFrame: The Loaded Data
    
    Returns:
        X_train: Training Data
        X_test: Testing Data
        y_train: Training Labels
        y_test: Testing Labels
    """
    try:
        data_cleaning = DataCleaningStrategy(df, DataPreprocessStrategy())
        processed_data = data_cleaning.handle_data()

        data_cleaning = DataCleaningStrategy(processed_data, DataSplitStrategy())
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        logging.info("Data cleaning Completed")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e
    