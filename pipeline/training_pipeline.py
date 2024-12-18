import os 
import sys 
from  src.components.data_ingestion import DataIngestion
from src.components.data_transformation import ColumnTransformation
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import Model_evaluation
from src.logging_config import logging
from zenml import step, pipeline

class training_pipeline:
    def start_data_ingestion(self):
        try:    
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.ingest_and_split(source_type='csv', 
        file_path='Data/dynamic_pricing.csv')
            return train_data_path, test_data_path
        except Exception as e:
            logging.error("There is prob in ingestion tp.")
            raise e
    
    def start_data_transformation(self, train_data_path, test_data_path):
        try:
            data_transformation = ColumnTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            return train_arr,test_arr
        except Exception as e:
            logging.error("There is prob in transformation tp.")
            raise e
            
    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer = ModelTrainer()
            model_trainer.initiate_model_training(train_arr, test_arr)
        except Exception as e:
            logging.error("There is prob in training tp.")
            raise e
    
    @pipeline(enable_cache=False)
    def start_training(self):
            try:
                train_data_path,test_data_path=self.start_data_ingestion()
                train_arr,test_arr=self.start_data_transformation(train_data_path,test_data_path)
                self.start_model_training(train_arr,test_arr)
            except Exception as e:
                raise e
if __name__ == "__main__":
    training_pipeline_obj = training_pipeline()
    training_pipeline_obj.start_training()
    