import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    A class responsible for ingesting data from a specified file path.
    Currently supports reading data from CSV and Excel files.
    """
    def __init__(self, data_path: str):
        """
        Initialises the file object with the path to the data.

        Args:
            data_path (str): The file path to the data source.
        """
        self.data_path = data_path

    def get_data(self):
        """
        Reads the data from the specified file path into a pandas DataFrame. 

        Return:
            pd.DataFrame : The loaded DataFrame.
            or
            ErrorMessage: "File extension unknown, try Ingesting file with .csv or .xslx extension."
        """
        logging.info(f"Ingesting data from {self.data_path}")
        if self.data_path.endswith(".csv"):
            return pd.read_csv(self.data_path)
        elif self.data_path.endswith(".xlsx"):
            return pd.read_excel(self.data_path)
        else:
            return ("File extension unknown, try Ingesting file with .csv or .xslx extension.")

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from data_path

    Args:
        data_path(str): specified path to the dataset

    Returns:
        pd.DataFrame: The loaded Data.
    
    Raises: 
        FileNotFoundError: If the file specified by the datapath does not exist.
        Other Excepttions: Other Exceptions aside from FileNotFoundError 
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except FileNotFoundError as e:
        logging.error(f"Error: File not found at {data_path}: {e}")
    except Exception as e:
        logging.error("error while ingesting the data: {e}")
        raise e