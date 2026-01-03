import os
import time
import warnings
from datetime import datetime

import joblib
import pandas as pd
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import save_params_model_with_best_params

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")


class BenchmarkPipeline:
    def __init__(self, random_state=42, models_dir=None):
        self.random_state = random_state
        self.DIST_MAP = {
            "randint": randint,
            "uniform": uniform,
            "loguniform": loguniform,
        }
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def build_preprocessor(self, preprocessing_file, n_neighbors=5):
        knn_cols = []
        mean_cols = []
        drop_cols = []

        for col, strategy in preprocessing_file.items():
            if strategy == 'knn':
                knn_cols.append(col)
            elif strategy == 'mean':
                mean_cols.append(col)
            elif strategy == 'delete':
                drop_cols.append(col)
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}")

        transformers = []
        if knn_cols:
            transformers.append(("knn_impute", KNNImputer(n_neighbors=n_neighbors), knn_cols))
        if mean_cols:
            transformers.append(("mean_impute", SimpleImputer(strategy="mean"), mean_cols))
        if drop_cols:
            transformers.append(("drop_cols", 'drop', drop_cols))

        return ColumnTransformer(transformers=transformers, remainder='passthrough')

    def get_model_class(self, model_name=None):
        models = {
            "LogisticRegression": lambda: LogisticRegression(random_state=self.random_state),
            "KNeighborsClassifier": lambda: KNeighborsClassifier(),
            "SVC": lambda: SVC(probability=True),
            "NaiveBayes": lambda: GaussianNB(),
            "DecisionTreeClassifier": lambda: DecisionTreeClassifier(random_state=self.random_state),
            "RandomForestClassifier": lambda: RandomForestClassifier(random_state=self.random_state),
            "XGBClassifier": lambda: XGBClassifier(random_state=self.random_state),
            "LGBMClassifier": lambda: LGBMClassifier(random_state=self.random_state),
        }

        if model_name is None:
            return models
        return models.get(model_name)

    def create_pipeline(self, model_name, preprocessing_file):
        model_cls = self.get_model_class(model_name)
        model = model_cls()

        preprocessor = self.build_preprocessor(preprocessing_file)

        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('scaler', 'passthrough'),
            ('sampler', 'passthrough'),
            ('clf', model)
        ])
        return pipe

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

    def _prepare_data(self, data):
        X = data.drop(columns="target")
        y = data["target"]
        return X, y

    def get_scalers_and_samplers_grid(self):
        return {
            "scaler": [
                StandardScaler(),
                MinMaxScaler(),
                "passthrough"
            ],
            "sampler": [
                "passthrough",
                RandomOverSampler(random_state=self.random_state),
                SMOTE(random_state=self.random_state),
                RandomUnderSampler(random_state=self.random_state)
            ]
        }

    def run_pipeline(self, data, model_name, model_file, preprocessing_file):
        X, y = self._prepare_data(data)
        pipe = self.create_pipeline(model_name, preprocessing_file)
        param_distributions = {
            **self.get_scalers_and_samplers_grid(),
            **self.get_param_distribution(model_file, model_name)
        }
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        start_time = time.time()
        search = RandomizedSearchCV(estimator=pipe, param_distributions=param_distributions, n_iter=5,
                                    scoring="roc_auc",
                                    cv=cv,
                                    n_jobs=-1,
                                    verbose=0,
                                    )
        search.fit(X, y)
        train_time = time.time() - start_time

        best_estimator = search.best_estimator_
        best_params = search.best_params_

        scaler = best_params["scaler"]
        sampler = best_params["sampler"]

        scaler_name = (
            type(scaler).__name__ if scaler != "passthrough" else "passthrough"
        )
        sampler_name = (
            type(sampler).__name__ if sampler != "passthrough" else "passthrough"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{scaler_name}_{sampler_name}_{timestamp}.joblib"
        model_path = os.path.join(self.models_dir, model_filename)
        joblib.dump(best_estimator, model_path)

        result = save_params_model_with_best_params(
            model=model_name,
            scaler=scaler_name,
            balancing_name=sampler_name,
            training_time=train_time,
            cv_roc_auc=search.best_score_,
            best_params=best_params,
            model_path=model_path,
        )
        return [result]

    def run_all_models(self, data, model_file, preprocessing_file):
        all_model_names = self.get_model_class().keys()
        all_results = []

        for model_name in all_model_names:
            print(f"\n{'=' * 50}")
            print(f"START PROCESSING MODEL: {model_name}")
            print(f"{'=' * 50}")

            results_for_model = self.run_pipeline(data, model_name, model_file, preprocessing_file)
            all_results.extend(results_for_model)

        return all_results

    def evaluate_model_on_valid_test(self, model_path, valid_df, has_target=False):
        model = joblib.load(model_path)
        X = valid_df.drop(columns="target") if has_target else valid_df

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        output_dir = "results/predictions"
        os.makedirs(output_dir, exist_ok=True)

        if has_target:
            y = valid_df["target"]
            return {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1": f1_score(y, y_pred),
                "roc_auc": roc_auc_score(y, y_proba)
            }
        else:
            df = pd.DataFrame({"prediction": y_pred, "probability": y_proba})
            df.to_csv(
                os.path.join(output_dir, f"{os.path.splitext(os.path.basename(model_path))[0]}_valid_predictions.csv"),
                index=False)
            return df
