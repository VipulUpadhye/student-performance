from pathlib import Path
import joblib
import sys
sys.path.append(str(Path(__file__).parent.resolve()))
from typing import Dict, Any

from sklearn.metrics import r2_score
from numpy.typing import ArrayLike

from src.exception import CustomException


def save_model(file_path, model_obj):
    '''
    Function to save all model objects (ML models and data preprocessors)
    '''
    try:
        dir_path = Path(file_path).parent.resolve()
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as file:
            joblib.dump(model_obj, file)
    except Exception as err:
        raise CustomException(err, sys)
    
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            return joblib.load(file)
    except Exception as err:
        raise CustomException(err, sys)
    
def model_eval(X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike, models_dict: Dict[str, Any]) -> Dict[str, float]:
    '''
    Function to evaluate the performance of given models on the test dataset
    '''
    try:
        report = {}

        for name, model in models_dict.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_r2_score = r2_score(y_test, y_test_pred)

            report[name] = test_r2_score
        
        return report
    
    except Exception as err:
        raise CustomException(err, sys)