from pathlib import Path
from utils import load_data
from pipeline import BenchmarkPipeline

root = Path.cwd().parent
data = load_data(f"{root}/data/valid.csv")  # zakładam pd.DataFrame
models_dir = f"{root}/results/models"

pipe = BenchmarkPipeline(models_dir=models_dir)
evaluate_valid_data = pipe.evaluate_model_on_valid_test(data, False)
evaluate_test_data = pipe.evaluate_model_on_valid_test(data, True)

evaluate_valid_data
evaluate_test_data