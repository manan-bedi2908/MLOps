import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluating our model
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: True Labels
            y_pred: Predicted Labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE...")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in Calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score...")
            r2 = mean_squared_error(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in Calculating R2 Score: {e}")
            raise e
        
