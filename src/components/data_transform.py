from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_model

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformConfig:
    preprocessor_obj_filepath = Path('artifacts').joinpath('preprocessor.joblib')


class DataTransform:
    def __init__(self):
        self.data_transform_config = DataTransformConfig()

    def get_data_transform_obj(self):
        '''
        Function for creating the data transformer objects.
        '''
        try:
            num_cols = ['writing_score', 'reading_score']
            cat_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info('Numerical columns transformation completed')

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info('Categorical columns transformation completed')

            preprocesor = ColumnTransformer(
                transformers = [
                    ('num_transform', num_pipeline, num_cols),
                    ('cat_transform', cat_pipeline, cat_cols)
                ]
            )
            logging.info('Preprocessor created for numerical and categorical columns')

            return preprocesor
        except Exception as err:
            raise CustomException(err, sys)
        
    def init_data_transform(self, train_path, test_path):
        try:
            logging.info('Read train and test datasets')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Generating data preprocessor object')
            preprocessor = self.get_data_transform_obj()

            target = 'math_score'

            df_train_feat = train_df.drop(columns=[target])
            df_train_label = train_df[target]

            df_test_feat = test_df.drop(columns=[target])
            df_test_label = test_df[target]

            logging.info('Applying data transformations to train and test sets')
            train_feat_transformed = preprocessor.fit_transform(df_train_feat, df_train_label)
            test_feat_transformed = preprocessor.transform(df_test_feat)

            train_feat_arr = np.c_[train_feat_transformed, df_train_label.values]
            test_feat_arr = np.c_[test_feat_transformed, df_test_label.values]

            logging.info('Saving the preprocessor object')
            save_model(file_path=self.data_transform_config.preprocessor_obj_filepath, model_obj=preprocessor)

            return(
                train_feat_arr,
                test_feat_arr,
                self.data_transform_config.preprocessor_obj_filepath
            )
        except Exception as err:
            raise CustomException(err, sys)