import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from zenml.steps import step
from abc import ABC, abstractmethod

# Singleton configuration class
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

@dataclass
class DataIngestionConfig(metaclass=SingletonMeta):
    """Singleton Configuration class for data ingestion."""
    artifacts_dir: Path = Path('artifacts')
    train_data_path: Path = artifacts_dir / 'train.csv'
    test_data_path: Path = artifacts_dir / 'test.csv'
    raw_data_path: Path = artifacts_dir / 'raw.csv'
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

# Abstract class for data ingestion
class DataIngestionStrategy(ABC):
    @abstractmethod
    def ingest_data(self):
        pass

# Concrete class for CSV ingestion
class CSVDataIngestion(DataIngestionStrategy):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def ingest_data(self):
        """Ingest data from a CSV file."""
        logging.info(f"Ingesting data from {self.file_path}")
        df = pd.read_csv(self.file_path)
        logging.info("Data ingestion from CSV completed.")
        return df

# Concrete class for future database ingestion (extendable)
class DatabaseDataIngestion(DataIngestionStrategy):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def ingest_data(self):
        """Ingest data from a database."""
        logging.info(f"Ingesting data from the database: {self.connection_string}")
        df = pd.DataFrame({'example': [1, 2, 3]})  # Simulated DataFrame
        logging.info("Data ingestion from database completed.")
        return df

# Factory to create the appropriate data ingestion strategy
class DataIngestionFactory:
    @staticmethod
    def get_ingestion_strategy(source_type: str, **kwargs) -> DataIngestionStrategy:
        if source_type == 'csv':
            return CSVDataIngestion(kwargs['file_path'])
        elif source_type == 'database':
            return DatabaseDataIngestion(kwargs['connection_string'])
        else:
            raise ValueError(f"Unknown source type: {source_type}")

# ZenML step class
@step
class DataIngestion:
    def __call__(self, source_type: str, **kwargs):
        """Executes data ingestion based on source type."""
        try:
            # Get the appropriate ingestion strategy from the factory
            ingestion_strategy = DataIngestionFactory.get_ingestion_strategy(source_type, **kwargs)
            df = ingestion_strategy.ingest_data()

            # Use the Singleton configuration
            config = DataIngestionConfig()

            # Save raw data
            logging.info(f"Saving raw data to {config.raw_data_path}")
            df.to_csv(config.raw_data_path, index=False)

            # Split data into train and test sets
            logging.info("Splitting data into train and test sets...")
            train_data, test_data = train_test_split(
                df, test_size=config.test_size, random_state=config.random_state
            )

            # Save train and test datasets
            train_data.to_csv(config.train_data_path, index=False)
            test_data.to_csv(config.test_data_path, index=False)
            logging.info(f"Train data saved to {config.train_data_path}")
            logging.info(f"Test data saved to {config.test_data_path}")

            return config.train_data_path, config.test_data_path

        except Exception as e:
            logging.error("Error occurred during data ingestion.", exc_info=True)
            raise e




data_ingestion = DataIngestion()

# Trigger the data ingestion step (from a CSV)
train_path, test_path = data_ingestion.initiate_data_ingestion(
    source_type='csv', file_path='Data/dynamic_pricing.csv'
)
tion_string'
