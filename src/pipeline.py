import time
import warnings
from itertools import product

from sklearn import set_config
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from config.experiment_config import param_distributions
from utils import create_pipeline, save_params_model_with_best_params, to_dataframe, prepare_data, MODELS, SCALERS, \
    SAMPLERS

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")


class Benchmark:
    def __init__(self, data, preprocessing_file, save_path):
        self.data = data
        self.preprocessing_file = preprocessing_file
        self.save_path = save_path

    def run_pipeline(self, model_name, scaler, sampler):
        X, y = prepare_data(self.data)

        pipe = create_pipeline(self.preprocessing_file)
        pipe.set_params(
            clf=MODELS[model_name],
            scaler=scaler,
            sampler=sampler
        )

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions[model_name],
            n_iter=100,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        start_time = time.time()
        search.fit(X, y)
        train_time = time.time() - start_time

        result = save_params_model_with_best_params(
            model=model_name,
            scaler=scaler if scaler == "passthrough" else type(scaler).__name__,
            balancing_name=sampler if sampler == "passthrough" else type(sampler).__name__,
            training_time=train_time,
            f1=search.best_score_,
            best_params=search.best_params_
        )
        return result


    def run(self):
        all_model_names = list(MODELS.keys())
        combinations = list(product(all_model_names, SCALERS.keys(), SAMPLERS.keys()))
        all_results = []
        for i, (model_name, scaler, sampler) in enumerate(combinations, start=1):
            print(f"Processing {i}/{len(combinations)} : {model_name} | {scaler} | {sampler}")
            result = self.run_pipeline(model_name, scaler, sampler)
            all_results.append(result)

        to_dataframe(all_results, self.save_path)


if __name__ == '__main__':
    from pathlib import Path
    from utils import load_config, load_data

    root = Path(__file__).parent
    preprocessing_file = load_config(f"{root}/config/preprocessing.yaml")
    data = load_data(f"{root}/../data/train.csv")
    bench = Benchmark(data, preprocessing_file, save_path="metrics")
    # bench.run_pipeline("LogisticRegression", "passthrough", "passthrough")
    bench.run()