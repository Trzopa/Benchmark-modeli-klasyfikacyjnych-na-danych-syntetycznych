from pathlib import Path
from utils import load_data, to_dataframe
from pipeline import BenchmarkPipeline

root = Path.cwd().parent
data = load_data(f"{root}/data/valid.csv")  # zakładam pd.DataFrame
models_dir = f"{root}/results/models"

pipe = BenchmarkPipeline(models_dir=models_dir)
evaluate_valid_data = pipe.evaluate_model_on_valid_test(models_dir, data, False)
evaluate_test_data = pipe.evaluate_model_on_valid_test(models_dir, data, True)
to_dataframe(evaluate_valid_data, "predictions")
to_dataframe(evaluate_test_data, "predictions")
