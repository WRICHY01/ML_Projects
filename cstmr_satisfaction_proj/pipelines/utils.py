import logging
import os

import pandas as pd

from src.data_cleaning import DataPreprocessStrategy, DataCleaningStrategy

module_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(module_path, "..", "dataset", "olist_customers_dataset.csv")

def get_data_for_test():
    try:
        df = pd.read_csv(file_path)
        df = df.sample(n = 1)
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaningStrategy(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["review_score"], axis = 1, inplace = True)
        result = df.T.to_json(orient = "split")
        return result


    except Exception as e:
        logging.error(e)
        raise e
