from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    '''
    Dataclass for defining the input variables in the data ingestion task.
    '''
    train_data_path: str = Path('artifacts').joinpath('train.csv')
    test_data_path: str = Path('artifacts').joinpath('test.csv')
    raw_data_path: str = Path('artifacts').joinpath('raw_data.csv')


class DataIngestion:
    def __init__(self):
        '''
        Fetch the data paths for the data ingestion task.
        '''
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingest(self):
        '''
        Ingest and read data from input source.
        '''
        logging.info('Started the data ingestion component')
        try:
            input_data_folder = Path(__file__).parent.parent.parent.joinpath('notebooks/data')
            df = pd.read_csv(input_data_folder.joinpath('stud.csv'))
            logging.info('Read the raw data in dataframe')

            Path(self.ingestion_config.train_data_path).parent.mkdir(exist_ok=True, parents=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

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
    obj = DataIngestion()
    obj.init_data_ingest()
