import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass                                # Decorator to declare a dataclass for data ingestion configuration
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")  # Default path for training data
    test_data_path: str = os.path.join('artifacts',"test.csv")    # Default path for testing data
    raw_data_path: str = os.path.join('artifacts', 'data.csv')    # Default path for raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initializing the data ingestion configuration
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Logging the entry of data ingestion method
        try:
            df = pd.read_csv('notebook\data\stud.csv')  # Reading the dataset as a dataframe
            logging.info("Read the dataset as dataframe.")  # Logging successful dataframe creation

            # Creating directory structure for train data path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving the dataframe as raw data
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("Train Test spit initiated")  # Logging initiation of train-test split
            
            # Splitting the dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving the train set
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)

            # Saving the test set
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info("Ingestion of the data is completed.")  # Logging completion of data ingestion

            return(
                self.ingestion_config.train_data_path,   # Returning the path of the train data
                self.ingestion_config.test_data_path,    # Returning the path of the test data
            )
        except Exception as e:
            raise CustomException(e,sys)   # Raising a custom exception if any error occurs
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array=train_array,test_array=test_array))




