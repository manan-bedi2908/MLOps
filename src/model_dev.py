import logging
from abc import ABC, abstractmethod

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