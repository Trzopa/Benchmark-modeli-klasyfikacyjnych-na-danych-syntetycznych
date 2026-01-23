from pathlib import Path
from utils import load_config, load_data, to_dataframe
from pipeline import BenchmarkPipeline

root = Path.cwd().parent
model_file = load_config("config/model.yaml")
preprocessing_file = load_config("config/preprocessing.yaml")
data = load_data(f"{root}/data/train.csv")
models_dir = f"{root}/results/models"
p = BenchmarkPipeline()
all_models = p.run_all_models(data, model_file, preprocessing_file, models_dir)
to_dataframe(all_models, "metrics")
