from dataclasses import dataclass
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_model, model_eval


@dataclass
class ModelTrainerConfig:
    trained_model_filepath = Path('artifacts').joinpath('trained_model.joblib').resolve()

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def init_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting dataset into features and label')
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'DecisionTree': DecisionTreeRegressor(),
                'LinearRegression': LinearRegression(),
                'KNN': KNeighborsRegressor(),
                'RandomForest': RandomForestRegressor(),
                'XGBoost': XGBRegressor()
            }

            model_report: dict = model_eval(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models_dict=models
            )

            # Get the best model name and score from model evaluation (model with highest R2 score on test set)
            best_model_name, best_model_score = max(model_report.items(), key=lambda item: item[1])

            # Get the best model object
            best_model = models[best_model_name]

            # Raise error if best model R2 score < 0.6
            if best_model_score < 0.6:
                raise CustomException('No best model found') 
            
            logging.info(f'Best model found. Model name: {best_model_name}, Test score: {best_model_score}')

            save_model(file_path=self.model_trainer_config.trained_model_filepath, model_obj=best_model)
            logging.info(f'Best performing model {best_model_name} saved to file')

            return (best_model_name, best_model_score)
        
        except Exception as err:
            raise CustomException(err, sys)