from abc import ABC, abstractmethod
import logging

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract Class Interface for leveraging Models during Training Phase.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the Model
        Args:
            X_train: Training Dataset
            y_train: Training Labels
        Returns:
            Trained Model Instance
        """
        pass


# class optimize(ABC):
#     pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the Model
        Args:
            X_train: Training Dataset
            y_train: Training Labels
        Returns:
        `   The Trained Model Instance
        """
        try:
            logging.info("Starting Linear Regression Model")
            self.model = LinearRegression(**kwargs)
            self.model.fit(X_train, y_train)
            logging.info("model training completed.")
            return self.model
        except Exception as e:
            logging.error(f"Error Training Model: {e}")
            raise e
        
    def predict(self, X_test):
        """
        Makes prediction on the input Dataset
        Args:
            X_test: Testing Dataset
        Returns:
            The predicted outcome from the unseen data(X_test)
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet, call train() first")
            return self.model.predict(X_test)
        except Exception as e:
            logging.error(f"Error Predicting using Model:{e}")
            raise e
