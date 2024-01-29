import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the Data Path
    """
    def __init__(self, data_path: str):
        """
        Args: 
            data_path: Path to the Data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting the Data from the Data Path
        """
        logging.info(f"Ingesting Data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the Data Path

    Args:
        data_path: Path to the Data

    Returns:
        pd.DataFrame: The Ingested Data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while Ingesting Data: {e}")
        raise e