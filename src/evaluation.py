import logging
from abc import ABC, abstractmethod
import numpy as np

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
