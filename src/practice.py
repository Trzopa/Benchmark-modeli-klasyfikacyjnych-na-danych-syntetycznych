from pathlib import Path
from utils import load_config, load_data, to_dataframe
from pipeline import Pipeline

root = Path.cwd().parent
model_file = load_config("config/model_params.yaml")
preprocessing_file = load_config("config/preprocessing.yaml")
data = load_data(f"{root}/data/train.csv")
p = Pipeline()
learn = p.run_pipeline(data, preprocessing_file, "LogisticRegression")
learn1 = p.run_pipeline_with_grid_search_cv(data, preprocessing_file, model_file, "LogisticRegression")

to_dataframe(learn, "learn")
