from abc import ABC, abstractmethod
import logging
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, root_mean_squared_error

class ModelEvaluator(ABC):
    """
    Abstract Class Interface for evaluating model
    """
    @abstractmethod
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(ModelEvaluator):
    """
    Evaluation Metric that uses Mean Squared Error to evaluate the Model.
    """
    def __init__(self):
        self.mse_score = None

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluates the model using Mean Squared Error(MSE) Metric
        Args:
            y_true(np.ndarray): true label from the unseen dataset.
            y_pred(np.ndarray): predicted label from the trained model.
        returns:
            The MSE metric result object
        """
        try:
            logging.info("Calculating MSE")
            self.mse_score = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {self.mse_score}")
            return self.mse_score

        except Exception as e:
            logging.error(f"Failed to Calculate MSE Score: {e}")
            raise e
        
class R2Score(ModelEvaluator):
    """
    Evaluation Metric that leverages r2_score to evaluate the Model.
    """
    def __init__(self):
        self.r2s_score = None

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluates the model using r2_score Metric
        Args:
            y_true(np.ndarray): true label from the unseen dataset.
            y_pred(np.ndarray): predicted label from the trained model.
        returns:
            The r2_score metric result object
        """
        try:
            logging.info("Calculating r2_score")
            self.r2s_score = r2_score(y_true, y_pred)
            logging.info(f"r2_score: {self.r2s_score}")
            return self.r2s_score
        
        except Exception as e:
            logging.error("Failed to Calculate r2_score: {e}")
            raise e

class RMSE(ModelEvaluator):
    """
    Evaluation Metric that leverages RMSE to evaluate Model performance.
    """
    def __init__(self):
        self.rmse_score = None

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluates the model using Root Mean Squared Error(RMSE) Metric
        Args:
            y_true(np.ndarray): true label from the unseen dataset.
            y_pred(np.ndarray): predicted label from the trained model.
        returns:
            The rmse metric result object
        """
        try:
            logging.info("Calculating RMSE")
            self.rmse_score = root_mean_squared_error(y_true, y_pred)
            logging.info(f"rmse_score: {self.rmse_score}")
            return self.rmse_score
        
        except Exception as e:
            logging.error(f"Failed to Calculate RMSE_score: {e}")
            raise e
    


