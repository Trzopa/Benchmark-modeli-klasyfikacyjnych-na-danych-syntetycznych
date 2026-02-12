import ast
import time
import warnings
from itertools import product
from pathlib import Path

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import load_config, load_data, save_params_model_with_evaluate_valid_data, \
    save_params_model_with_evaluate_test_data
from utils import save_params_model_with_best_params

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")


class BenchmarkPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def __get_dist_params(self):
        return {
            "randint": randint,
            "uniform": uniform,
            "loguniform": loguniform,
        }

    def __get_param_distribution(self, config_name, model_name):
        model_config = config_name[model_name]
        param_distributions = {}

        for param_name, spec in model_config.items():
            # 1. Proste listy
            if isinstance(spec, list):
                param_distributions[param_name] = spec

            # 2. NOWOŚĆ: Obsługa "value" (np. random_state: value: 42)
            elif isinstance(spec, dict) and "value" in spec:
                param_distributions[param_name] = [spec["value"]]

            # 3. Rozkłady statystyczne
            elif isinstance(spec, dict) and "distribution" in spec:
                dist_name = spec["distribution"]
                dist_cls = self.__get_dist_params()[dist_name]

                if dist_name == "randint":
                    param_distributions[param_name] = dist_cls(low=spec["low"], high=spec["high"])
                elif dist_name == "uniform":
                    param_distributions[param_name] = dist_cls(loc=spec["loc"], scale=spec["scale"])
                elif dist_name == "loguniform":
                    param_distributions[param_name] = dist_cls(spec["low"], spec["high"])

        return param_distributions

    def __get_model(self, model_name=None):
        models = {
            "LogisticRegression": LogisticRegression(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "SVC": SVC(probability=True),
            "NaiveBayes": GaussianNB(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier(),
            "LGBMClassifier": LGBMClassifier(),
        }

        if model_name is None:
            return models
        return models.get(model_name)

    def __build_preprocessor(self, preprocessing_file, n_neighbors=5):
        knn_cols = []
        mean_cols = []
        drop_cols = []

        # Categorize columns by their imputation strategy
        for col, strategy in preprocessing_file.items():
            if strategy == 'knn':
                knn_cols.append(col)
            elif strategy == 'mean':
                mean_cols.append(col)
            elif strategy == 'delete':
                drop_cols.append(col)
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}")

        # Build list of transformers based on configured strategies
        transformers = []
        if knn_cols:
            transformers.append(("knn_impute", KNNImputer(n_neighbors=n_neighbors), knn_cols))
        if mean_cols:
            transformers.append(("mean_impute", SimpleImputer(strategy="mean"), mean_cols))
        if drop_cols:
            transformers.append(("drop_cols", 'drop', drop_cols))

        return ColumnTransformer(transformers=transformers, remainder='passthrough')

    def create_pipeline(self, model_name, preprocessing_file):
        model = self.__get_model(model_name)
        preprocessor = self.__build_preprocessor(preprocessing_file)

        # Pipeline steps: preprocessing -> scaling -> sampling -> classification
        # 'passthrough' placeholders will be replaced during hyperparameter search
        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('scaler', 'passthrough'),
            ('sampler', 'passthrough'),
            ('clf', model)
        ])
        return pipe

    def __prepare_data(self, data):
        if "target" in data.columns:
            X = data.drop(columns="target")
            y = data["target"]
            return X, y
        else:
            return data, None



    def __get_scalers_and_samplers_grid(self):
        return {
            "scaler": [
                "passthrough",  # No scaling
                StandardScaler(),
                MinMaxScaler(),
            ],
            "sampler": [
                "passthrough",  # No resampling
                RandomOverSampler(random_state=self.random_state),
                RandomUnderSampler(random_state=self.random_state),
                SMOTE(random_state=self.random_state),
            ]
        }

    def run_pipeline(self, data, model_name, model_file, preprocessing_file, scaler_obj, sampler_obj, ):
        # Prepare features and target
        X, y = self.__prepare_data(data)

        pipe = self.create_pipeline(model_name, preprocessing_file)

        pipe.set_params(scaler=scaler_obj, sampler=sampler_obj)
        param_distributions = self.__get_param_distribution(model_file, model_name)
        scaler_name = type(scaler_obj).__name__ if scaler_obj != "passthrough" else "passthrough"
        sampler_name = type(sampler_obj).__name__ if sampler_obj != "passthrough" else "passthrough"

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions,
            n_iter=100,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        start_time = time.time()
        search.fit(X, y)
        train_time = time.time() - start_time

        result = save_params_model_with_best_params(
            model=model_name,
            scaler=scaler_name,
            balancing_name=sampler_name,
            training_time=train_time,
            cv_roc_auc=search.best_score_,
            best_params=search.best_params_,
        )
        return [result]

    def run_all_models(self, data, model_file, preprocessing_file):
        all_model_names = list(self.__get_model().keys())
        grid_options = self.__get_scalers_and_samplers_grid()

        all_results = []
        combinations = list(product(all_model_names, grid_options["scaler"], grid_options["sampler"]))

        for model_name, scaler_obj, sampler_obj in combinations:
            print(f"Przetwarzanie {len(all_results) + 1}/96: {model_name}")

            result = self.run_pipeline(
                data, model_name, model_file, preprocessing_file, scaler_obj, sampler_obj
            )
            all_results.extend(result)

        return all_results


# TODO: testowanie  doanie printow w ostatniej metodzie aby wyspitlilo ile tych parametrow przeszlo
# TODO: zrobic benchmark dla valid i testu
# TODO: zrobic report
# TODO:


if __name__ == "__main__":
    root = Path.cwd().parent
    data = load_data(f"{root}/results/metrics/results_20260127_112530.csv")
    data_train = load_data(f"{root}/data/train.csv")
    data_test = load_data(f"{root}/data/test.csv")
    data_valid = load_data(f"{root}/data/valid.csv")
    root = Path.cwd().parent
    model_file = load_config("config/model.yaml")
    preprocessing_file = load_config("config/preprocessing.yaml")
    b = BenchmarkPipeline()
    ve = b.evaluate_to_valid_data(data_train, data_valid, data, preprocessing_file)
    # te = b.evaluate_to_test_data(data_train, data_test, data, preprocessing_file)
    print(ve)
    # Poprawna nazwa to dtypes
