from pathlib import Path
from utils import load_config, load_data, to_dataframe
from pipeline import Pipeline

root = Path.cwd().parent
model_file = load_config("config/model.yaml")
preprocessing_file = load_config("config/preprocessing.yaml")
data = load_data(f"{root}/data/test.csv")
p = Pipeline()
pdata = p.run_pipeline(data, model_file, preprocessing_file, "LogisticRegression")

# all_models = p.run_all_models(data, preprocessing_file, model_file, True)
# to_dataframe(all_models, "predictions")
