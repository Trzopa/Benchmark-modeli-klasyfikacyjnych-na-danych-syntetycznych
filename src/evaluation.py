import ast
import time
import warnings

from sklearn import set_config

from utils import prepare_data, \
    MODELS, create_pipeline, SCALERS, SAMPLERS, evaluate_test, evaluate_valid

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def __parse_best_params(params_str):
    clean_str = params_str.replace('np.float64(', '').replace(')', '')
    return ast.literal_eval(clean_str)


def get_configs(results_df):
    all_configs = []
    for _, row in results_df.iterrows():
        params = __parse_best_params(row['best_params'])

        configs = {
            'model': row['model'],
            'scaler': row['scaler'],
            'sampler': row['balancing_name'],
            'params': params
        }
        all_configs.append(configs)
    return all_configs


class ModelEvaluator:
    def __init__(self, train_data, valid_data, preprocessing_file, test_data, results_df):
        self.results_df = results_df
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.preprocessing_file = preprocessing_file

    def evaluate_to_valid_data(self):
        configs = get_configs(self.results_df)

        X_train, y_train = prepare_data(self.train_data)
        X_valid, _ = prepare_data(self.valid_data)

        results = []

        for config in configs:
            pipe = create_pipeline(self.preprocessing_file)

            params = config["params"].copy()

            pipe.steps = [
                ("preprocessor", pipe.named_steps["preprocessor"]),
                ("scaler", SCALERS[config["scaler"]]),
                ("sampler", SAMPLERS[config["sampler"]]),
                ("clf", MODELS[config["model"]]),
            ]

            pipe.set_params(**params)

            start_time = time.time()
            pipe.fit(X_train, y_train)
            duration = time.time() - start_time

            y_pred = pipe.predict(X_valid)
            y_proba = pipe.predict_proba(X_valid)[:, 1]

            result = evaluate_valid(y_pred, y_proba, config, duration)
            results.append(result)

        return results

    def evaluate_to_test_data(self):
        configs = get_configs(self.results_df)
        X_train, y_train = prepare_data(self.train_data)
        X_test, y_test = prepare_data(self.test_data)
        results = []
        for config in configs:
            pipe = create_pipeline(self.preprocessing_file)

            params = config["params"].copy()

            pipe.steps = [
                ("preprocessor", pipe.named_steps["preprocessor"]),
                ("scaler", SCALERS[config["scaler"]]),
                ("sampler", SAMPLERS[config["sampler"]]),
                ("clf", MODELS[config["model"]]),
            ]

            pipe.set_params(**params)

            start_time = time.time()
            pipe.fit(X_train, y_train)
            duration = time.time() - start_time

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            result = evaluate_test(y_test, y_pred, y_proba, config, duration)
            results.append(result)

        return results

        return results


if __name__ == '__main__':
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
