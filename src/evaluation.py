import time
import warnings

from sklearn import set_config

from utils import prepare_data, \
    MODELS, create_pipeline, SCALERS, SAMPLERS, evaluate_test, evaluate_valid, get_configs

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")


class ModelEvaluator:
    def __init__(self, train_data, valid_data, preprocessing_file, test_data, results_df):
        self.results_df = results_df
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.preprocessing_file = preprocessing_file

    def _evaluate(self, X_train, y_train, X_eval, y_eval, evaluate_fn):
        configs = get_configs(self.results_df)
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

            y_pred = pipe.predict(X_eval)

            y_proba = pipe.predict_proba(X_eval)[:, 1]

            result = evaluate_fn(y_eval, y_pred, y_proba, config, duration)
            results.append(result)

        return results

    def evaluate_to_valid_data(self):
        X_train, y_train = prepare_data(self.train_data)
        X_valid, y_valid = prepare_data(self.valid_data)

        return self._evaluate(
            X_train, y_train,
            X_valid, y_valid,
            evaluate_valid
        )

    def evaluate_to_test_data(self):
        X_train, y_train = prepare_data(self.train_data)
        X_test, y_test = prepare_data(self.test_data)

        return self._evaluate(
            X_train, y_train,
            X_test, y_test,
            evaluate_test
        )
