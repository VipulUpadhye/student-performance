from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.exception import CustomException
from src.utils import load_model

import pandas as pd


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features: pd.DataFrame):
        try:
            artifacts_folder_path = Path(__file__).parent.parent.parent.joinpath('artifacts').resolve()
            model_path = artifacts_folder_path.joinpath('trained_model.joblib')
            preprocessor_path = artifacts_folder_path.joinpath('preprocessor.joblib')
            
            trained_model = load_model(file_path=model_path)
            preprocessor = load_model(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = trained_model.predict(data_scaled)

            return preds
        except Exception as err:
            raise CustomException(err, sys)

class CustomData:
    '''
    Input data class to receive from HTML page for prediction pipeline
    '''
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
        ) -> None:
        '''
        Constructor for mapping the input data from the home.html form to the class variables
        '''
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_df(self) -> pd.DataFrame:
        '''
        Convert the input data from the input form home.html to a dataframe for model prediction
        '''
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as err:
            raise CustomException(err, sys)

