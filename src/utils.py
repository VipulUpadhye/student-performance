from pathlib import Path
import joblib
import sys
sys.path.append(str(Path(__file__).parent.resolve()))

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