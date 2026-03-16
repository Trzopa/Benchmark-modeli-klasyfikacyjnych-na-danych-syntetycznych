from pathlib import Path
from utils import load_config, load_data
from pipeline import Benchmark

root = Path(__file__).parent
model_file = load_config(f"{root}/config/model.yaml")
preprocessing_file = load_config(f"{root}/config/preprocessing.yaml")
data = load_data(f"{root}/../data/train.csv")
bench = Benchmark(data, model_file, preprocessing_file, save_path="metrics")
bench.run()
