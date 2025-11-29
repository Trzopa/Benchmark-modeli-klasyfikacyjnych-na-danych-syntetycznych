import inspect
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import uniform
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import load_config, load_data, save_params_model


class Pipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.DIST_MAP = {"randint": randint, "uniform": uniform}

    def preprocessing_data(self, data, preprocessing_config):
        mean_cols = [col for col, method in preprocessing_config.items() if method == "mean"]
        knn_cols = [col for col, method in preprocessing_config.items() if method == "knn"]
        cols_to_delete = [col for col, method in preprocessing_config.items() if method == "delete"]
        data = data.drop(columns=cols_to_delete)
        if mean_cols:
            mean_inputer = SimpleImputer(strategy="mean")
            data[mean_cols] = mean_inputer.fit_transform(data[mean_cols])
        if knn_cols:
            knn_inputer = KNNImputer(n_neighbors=5)
            data[knn_cols] = knn_inputer.fit_transform(data[knn_cols])
        return data

    def apply_balancing(self, X, y):
        balanced = SMOTE(random_state=self.random_state)
        X_balanced, y_balanced = balanced.fit_resample(X, y)
        return X_balanced, y_balanced

    def get_model_class(self, model_name=None):
        models = {
            "LogisticRegression": LogisticRegression,
            "KNeighborsClassifier": KNeighborsClassifier,
            "SVC": SVC,
            "NaiveBayes": GaussianNB,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "XGBClassifier": XGBClassifier,
            "LGBMClassifier": LGBMClassifier,
        }

        if model_name is None:
            return models

        return models.get(model_name, None)

    def get_param_distribution(self, config_name, model_name):
        model_config = config_name[model_name]
        param_distributions = {}

        for param_name, spec in model_config.items():
            if isinstance(spec, list):
                param_distributions[param_name] = spec

            elif isinstance(spec, dict) and "distribution" in spec:
                dist_name = spec["distribution"]
                dist_cls = self.DIST_MAP[dist_name]

                if dist_name == "randint":
                    param_distributions[param_name] = dist_cls(
                        low=spec["low"], high=spec["high"]
                    )
                elif dist_name == "uniform":
                    param_distributions[param_name] = dist_cls(
                        loc=spec["loc"], scale=spec["scale"]
                    )

        return param_distributions

    def create_pipeline_with_scaler(self, model_cls, scaler=None):
        steps = []
        if scaler:
            steps.append(("scaler", scaler))

        model_kwargs = {}

        if 'random_state' in inspect.signature(model_cls.__init__).parameters:
            model_kwargs['random_state'] = self.random_state

        steps.append(("clf", model_cls(**model_kwargs)))

        return SklearnPipeline(steps)

    def grid_search_cv(self, X, y, pipe, param_dist):
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1_score': make_scorer(f1_score, average='weighted'),
        }

        start_time = time.time()
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=5,
            cv=5,
            scoring=scorers,
            refit='f1_score',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X, y)
        end_time = time.time()
        duration = end_time - start_time
        best_index = search.best_index_
        results = search.cv_results_
        return {
            'duration': duration,
            'best_params': search.best_params_,
            'mean_test_accuracy': results['mean_test_accuracy'][best_index],
            'mean_test_precision': results['mean_test_precision'][best_index],
            'mean_test_recall': results['mean_test_recall'][best_index],
            'mean_test_f1_score': results['mean_test_f1_score'][best_index],
        }

    def train_model(self, X, y, pipe, model_name, scaler_name):
        start_time = time.time()
        pipe.fit(X, y)
        stop_time = time.time()
        time_training = stop_time - start_time
        y_pred = pipe.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        formatted_result = save_params_model(
            model=model_name,
            scaler=scaler_name,
            training_time=time_training,
            accuracy_score_val=accuracy,
            precision_score_val=precision,
            recall_score_val=recall,
            f1_score_val=f1,
        )

        return formatted_result

    def _prepare_data_and_pipelines(self, data, preprocessing_file, model_name):
        data_processed = self.preprocessing_data(data, preprocessing_file)
        X_processed = data_processed.drop(columns=["target"])
        y = data_processed["target"]

        X_balanced, y_balanced = self.apply_balancing(X_processed, y)

        model_cls = self.get_model_class(model_name)
        pipelines_to_test = {
            "StandardScaler": self.create_pipeline_with_scaler(model_cls, scaler=StandardScaler()),
            "MinMaxScaler": self.create_pipeline_with_scaler(model_cls, scaler=MinMaxScaler()),
            "None": self.create_pipeline_with_scaler(model_cls, scaler=None),
        }

        return X_balanced, y_balanced, pipelines_to_test

    def run_pipline(self, data, preprocessing_file, model_file, model_name):

        X_balanced, y_balanced, pipelines_to_test = self._prepare_data_and_pipelines(
            data, preprocessing_file, model_name
        )

        param_dist = self.get_param_distribution(model_file, model_name)
        all_results_list = []

        for scaler_name, pipe_with_scaler in pipelines_to_test.items():
            print(f"Trening i tuning z {scaler_name} scalerem...")

            best_results = self.grid_search_cv(X_balanced, y_balanced, pipe_with_scaler, param_dist)

            formatted_result = save_params_model(
                model=model_name,
                scaler=scaler_name,
                training_time=best_results['duration'],
                accuracy_score_val=best_results['mean_test_accuracy'],
                precision_score_val=best_results['mean_test_precision'],
                recall_score_val=best_results['mean_test_recall'],
                f1_score_val=best_results['mean_test_f1_score'],
                best_params=best_results['best_params']
            )

            all_results_list.append(formatted_result)

        return all_results_list

    def run_simple_pipeline(self, data, preprocessing_file, model_name):

        X_balanced, y_balanced, pipelines_to_test = self._prepare_data_and_pipelines(
            data, preprocessing_file, model_name
        )

        all_results_list = []

        for scaler_name, pipe_with_scaler in pipelines_to_test.items():
            print(f"Trening prostego modelu z {scaler_name} scalerem...")

            result = self.train_model(
                X_balanced,
                y_balanced,
                pipe_with_scaler,
                model_name,
                scaler_name
            )

            all_results_list.append(result)

        return all_results_list

    def run_all_models(self, data, preprocessing_file, model_file, use_grid_search=True):

        all_model_names = self.get_model_class().keys()

        all_results = []

        for model_name in all_model_names:
            print(f"\n{'=' * 50}")
            print(f"ROZPOCZĘCIE PRZETWARZANIA MODELU: {model_name}")
            print(f"{'=' * 50}")

            if use_grid_search:
                results_for_model = self.run_pipline(data, preprocessing_file, model_file, model_name)
            else:

                results_for_model = self.run_simple_pipeline(data, preprocessing_file, model_name)

            all_results.extend(results_for_model)

        return all_results

    def to_dataframe(self, results_list):
        df = pd.DataFrame(results_list)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        file_path = os.path.join("results", f"results_{timestamp}.csv")

        df.to_csv(file_path, index=False)
        print(f"Wyniki zapisane do: {file_path}")
        return df

if __name__ == "__main__":
    root = Path.cwd().parent
    model_file = load_config("config/models.yaml")
    preprocessing_file = load_config("config/preprocessing.yaml")
    data = load_data("/Users/trzopa/Benchmark-modeli-klasyfikacyjnych-na-danych-syntetycznych/data/train.csv")
    p = Pipeline()
    pdata = p.preprocessing_data(data, preprocessing_file)
    all_models = p.run_all_models(data, preprocessing_file, model_file, False)
    all_models
    results_df = p.to_dataframe(all_models)
