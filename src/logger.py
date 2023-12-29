import logging
from pathlib import Path
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'
logs_path = Path.cwd().joinpath('logs')
logs_path.mkdir(exist_ok=True, parents=True)

LOG_FILEPATH = logs_path.joinpath(LOG_FILE)

logging.basicConfig(
    filename=LOG_FILEPATH,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO, 
)