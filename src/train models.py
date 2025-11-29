from pathlib import Path

import uniform
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline  # Zmieniono nazwę, żeby uniknąć konfliktu

from utils import load_config, load_data


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

    def get_model_class(self, model_name):
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

    def grid_search_cv(self, X, y, pipe, param_dist):
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=5,
            cv=5,
            scoring='f1',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X, y)
        return search.best_params_

    def create_pipeline_with_scaler(self, model_cls, scaler=None):
        steps = []
        if scaler:
            steps.append(("scaler", scaler))
        steps.append(("clf", model_cls(random_state=self.random_state)))
        return SklearnPipeline(steps)

    def run_pipline(self, data, preprocessing_file, model_file, model_name):
        data_processed = self.preprocessing_data(data, preprocessing_file)
        X_processed = data_processed.drop(columns=["target"])
        y = data_processed["target"]
        X_balanced, y_balanced = self.apply_balancing(X_processed, y)

        param_dist = self.get_param_distribution(model_file, model_name)
        model_cls = self.get_model_class(model_name)
        results = {}
        pipelines_to_test = {
            "standard": self.create_pipeline_with_scaler(model_cls, scaler=StandardScaler()),
            "minmax": self.create_pipeline_with_scaler(model_cls, scaler=MinMaxScaler()),
            "none": self.create_pipeline_with_scaler(model_cls, scaler=None),
        }

        for scaler_name, pipe_with_scaler in pipelines_to_test.items():
            print(f"Trening i tuning z {scaler_name} scalerem...")
            best_model = self.grid_search_cv(X_balanced, y_balanced, pipe_with_scaler, param_dist)
            results[scaler_name] = best_model

        return results


# poprawić pipline
# wywolac dla kazdego modelu
# zapisac wyniki
#

if __name__ == "__main__":
    root = Path.cwd().parent
    model_file = load_config("config/models.yaml")
    preprocessing_file = load_config("config/preprocessing.yaml")
    data = load_data("/Users/trzopa/Benchmark-modeli-klasyfikacyjnych-na-danych-syntetycznych/data/train.csv")
    p = Pipeline()
    pdata = p.preprocessing_data(data, preprocessing_file)
    LogisticRegression = p.run_pipline(data, preprocessing_file, model_file, "LogisticRegression")
    print(LogisticRegression)
