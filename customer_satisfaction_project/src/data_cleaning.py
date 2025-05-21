import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessingStrategy(ABC):
    """
    Abstract class Interface for defining strategy of handling data.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        pass

class DataPreprocessStrategy(DataPreprocessingStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data = data.drop(
                [
                "customer_zip_code_prefix",
                "order_item_id",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp"         
                ], axis = 1)
            
            data = data.drop_duplicates()

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace = True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace = True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace = True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace = True)
            data["review_comment_message"].fillna("No review", inplace = True)

            data = data.select_dtypes(include=[np.number])

            return data

        except Exception as e:
            logging.error(f"Failed to preprocess data: {e}")
            raise e

class DataSplitStrategy(DataPreprocessingStrategy):
    """
    Strategy for splitting dataset in train and test set
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        try:
            X = data.drop(columns = ["review_score"], axis = 1)
            y = data["review_score"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Failed to Split the Dataset: {e}")
            raise e

class DataCleaningStrategy(DataPreprocessingStrategy):
    """
    Class for cleaning data which preprocesses the data and splits the dataset into
    training and test sets
    """
    def __init__(self, data: pd.DataFrame, strategy: DataPreprocessingStrategy):
        
        self.data = data
        self.strategy = strategy
    try:
        def handle_data(self):
            """
            Handles Data
            """
            return self.strategy.handle_data(self.data)
    except Exception as e:
        logging.error(f"Error in handling data: {e}")
        raise e


if __name__ == "__main__":
    data_df = pd.read_csv(r"\my_projects\customer_satisfaction_project\dataset\olist_customers_dataset.csv")
    data_clean_test = DataPreprocessStrategy().handle_data(data_df)
    print(data_clean_test.columns)
    data_split_test = DataSplitStrategy().handle_data(data_clean_test)
    print(data_split_test)

    


