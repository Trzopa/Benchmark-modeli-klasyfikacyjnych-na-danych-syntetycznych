from pathlib import Path
from utils import load_data, to_dataframe, load_config
from evaluation import ModelEvaluator

root = Path.cwd().parent

data = load_data(f"{root}/results/metrics/results_20260224_222500.csv")
data_train = load_data(f"{root}/data/train.csv")
data_test = load_data(f"{root}/data/test.csv")
data_valid = load_data(f"{root}/data/valid.csv")
root = Path.cwd().parent
preprocessing_file = load_config("config/preprocessing.yaml")
b = ModelEvaluator(data_train, data_valid, preprocessing_file, data_test, data)
evaluate_valid_data = b.evaluate_to_valid_data()
to_dataframe(evaluate_valid_data, "predictions")
evaluate_test_data = b.evaluate_to_test_data()
to_dataframe(evaluate_test_data, "predictions")
