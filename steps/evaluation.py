import logging
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluates the model on the Ingested Data
    Args:
        df: The Ingested Data
    """
    pass