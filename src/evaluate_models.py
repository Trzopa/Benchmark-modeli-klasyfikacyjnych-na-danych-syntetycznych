from datetime import datetime

from pathlib import Path

from utils import load_data
from utils import predict_valid, load_joblib

root = Path.cwd().parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{root}/results/predictions/all_models_{timestamp}.csv"
all_models_file = load_joblib(f"{root}/results/models/all_models_*.pkl")
data_valid = load_data(f"{root}/data/valid.csv")

predict_valid(data_valid, all_models_file, output_path)
