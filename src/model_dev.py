import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract Class for all Models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the Model
        Args:
            X_train: Training Data
            y_train: Train Labels
        Returns: 
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the Model
        Args:
            X_train: Training Data
            y_train: Train Labels
        Returns: 
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training Completed...")
            return reg
        except Exception as e:
            logging.error(f"Error in Training Model: {e}")
            raise e

