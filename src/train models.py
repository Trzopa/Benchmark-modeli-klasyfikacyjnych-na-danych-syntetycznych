from pathlib import Path
from utils import load_config, load_data, to_dataframe
from pipeline import Pipeline

root = Path.cwd().parent
model_file = load_config("config/model.yaml")
preprocessing_file = load_config("config/preprocessing.yaml")
data = load_data(f"{root}/data/train.csv")
p = Pipeline()

all_models = p.run_all_models(data, model_file, preprocessing_file)
to_dataframe(all_models, "metrics")
