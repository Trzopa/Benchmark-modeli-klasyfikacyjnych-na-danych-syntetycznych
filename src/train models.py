from pathlib import Path
from utils import load_config, load_data, to_dataframe
from pipeline import Pipeline

root = Path.cwd().parent
model_file = load_config("config/models.yaml")
preprocessing_file = load_config("config/preprocessing.yaml")
data = load_data(f"{root}/data/train.csv")
p = Pipeline()
pdata = p.preprocessing_data(data, preprocessing_file)

all_models = p.run_all_models(data, preprocessing_file, model_file, False)
to_dataframe(all_models, "metrics")
