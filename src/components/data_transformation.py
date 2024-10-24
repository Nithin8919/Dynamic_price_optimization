import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from abc import ABC, abstractmethod
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from dataclasses import dataclass
from typing import List
from sklearn.impute import SimpleImputer
import logging
import joblib

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    
class PreprocessorStrategy(ABC):
    @abstractmethod
    def preprocessor(self, df: pd.DataFrame):
        pass
    
class ColumnTransformation(PreprocessorStrategy):
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def preprocessor(self, df: pd.DataFrame):
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            logging.info(f"The categorical columns are {categorical_cols}")
            
            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            # For numerical columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            logging.info(f"The numerical columns are {numerical_cols}")
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Combine pipelines for numeric and categorical columns
            transform = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )
            
            return transform
        
        except Exception as e:
            logging.error("Exception occurred in preprocessing")
            raise e
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Get preprocessing pipeline
            processing = self.preprocessor(train_df)
            
            target_column = 'Historical_Cost_of_Ride'
            
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info("Applying preprocessing object on training and testing dataframes")

            # Apply fit_transform on training data and transform on test data
            input_feature_train_arr = processing.fit_transform(input_feature_train_df)
            input_feature_test_arr = processing.transform(input_feature_test_df)
            
            # Concatenating input and target features
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            # Save the preprocessing object
            self.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=processing
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.error("Exception occurred in the transformation step")
            raise e
    
    @staticmethod
    def save_object(file_path, obj):
        """Save the preprocessing object to a file."""
        try:
            with open(file_path, 'wb') as file:
                joblib.dump(obj, file)
            logging.info(f"Preprocessing object saved to {file_path}")
        except Exception as e:
            logging.error("Failed to save the preprocessing object.")
            raise e

