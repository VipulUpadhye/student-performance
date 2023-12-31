from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transform import DataTransformConfig, DataTransform
from src.components.model_train import ModelTrainerConfig, ModelTrainer


@dataclass
class DataIngestionConfig:
    '''
    Dataclass for defining the input variables in the data ingestion task.
    '''
    train_data_path = Path('artifacts').joinpath('train.csv')
    test_data_path = Path('artifacts').joinpath('test.csv')
    raw_data_path = Path('artifacts').joinpath('raw_data.csv')


class DataIngestion:
    def __init__(self):
        '''
        Fetch the data paths for the data ingestion task.
        '''
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingest(self):
        '''
        Function for performing the data ingestion task.

        Steps:
        1. Ingest and read data from input source.
        2. Split input data into train and test sets.
        3. Save the train and test data to CSV.
        '''
        logging.info('Started the data ingestion component')
        try:
            # Read the input data
            input_data_folder = Path(__file__).parent.parent.parent.joinpath('notebooks/data')
            df = pd.read_csv(input_data_folder.joinpath('stud.csv'))
            logging.info('Read the raw data in dataframe')

            # Generate the "artifacts" folder for storing the output data files
            Path(self.ingestion_config.train_data_path).parent.mkdir(exist_ok=True, parents=True)

            # Save the raw data to CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split raw data into train and test sets, and save the datasets to CSV
            logging.info('Train-Test split started')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=10)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as err:
            logging.info('Error in data ingestion task!!')
            raise CustomException(err, sys)


if __name__ == '__main__':
    data_ingest = DataIngestion()
    train_data_path, test_data_path = data_ingest.init_data_ingest()

    data_transform = DataTransform()
    train_arr, test_arr, _ = data_transform.init_data_transform(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    best_model_name, best_score = model_trainer.init_model_trainer(train_arr, test_arr)
    print(f'Best model is {best_model_name} with R2 score of {best_score} on test set')