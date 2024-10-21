import pandas as pd
import zenml
import logging as logging
import os
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """Data Ingestion Config"""
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path :str = os.path.join("artifacts",'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestionconfig = DataIngestionConfig
        
    def initiate_data_ingestion(self):
        """Initiate data ingestion"""
        try:
            df = pd.read_csv('Data/dynamic_pricing.csv')
            logging.info('Data ingestion is initiated !')
            
            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path), exist_ok= True)
            df.to_csv(self.ingestionconfig.raw_data_path,index=False,header=True)
            
            logging.info("train test split initiated")
            train_data, test_data = train_test_split(df,test_size=0.2,random_state=42)
            
            logging.info("ingestion is completed")
            
        
            return (
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )
        except Exception as e:
            raise e
            
            
            
