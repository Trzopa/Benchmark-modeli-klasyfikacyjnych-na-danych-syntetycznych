import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import save_params_model_with_best_params, save_machine_learning_model

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")


class BenchmarkPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def _get_dist_params(self):
        return {
            "randint": randint,
            "uniform": uniform,
            "loguniform": loguniform,
        }

    def get_param_distribution(self, config_name, model_name):
        model_config = config_name[model_name]
        param_distributions = {}

        for param_name, spec in model_config.items():
            # If spec is a list, use it directly (discrete choices)
            if isinstance(spec, list):
                param_distributions[param_name] = spec

            # If spec defines a distribution, create scipy stats object
            elif isinstance(spec, dict) and "distribution" in spec:
                dist_name = spec["distribution"]
                dist_cls = self._get_dist_params()[dist_name]

                if dist_name == "randint":
                    param_distributions[param_name] = dist_cls(
                        low=spec["low"], high=spec["high"]
                    )
                elif dist_name == "uniform":
                    param_distributions[param_name] = dist_cls(
                        loc=spec["loc"], scale=spec["scale"]
                    )

        return param_distributions

    def get_model_class(self, model_name=None):
        models = {
            # "LogisticRegression": lambda: LogisticRegression(random_state=self.random_state),
            # "KNeighborsClassifier": lambda: KNeighborsClassifier(),
            # "SVC": lambda: SVC(probability=True),
            # "NaiveBayes": lambda: GaussianNB(),
            # "DecisionTreeClassifier": lambda: DecisionTreeClassifier(random_state=self.random_state),
            # "RandomForestClassifier": lambda: RandomForestClassifier(random_state=self.random_state),
            # "XGBClassifier": lambda: XGBClassifier(random_state=self.random_state),
            "LGBMClassifier": lambda: LGBMClassifier(random_state=self.random_state),
        }

        if model_name is None:
            return models
        return models.get(model_name)

    def build_preprocessor(self, preprocessing_file, n_neighbors=5):
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
        model_cls = self.get_model_class(model_name)
        model = model_cls()

        preprocessor = self.build_preprocessor(preprocessing_file)

        # Pipeline steps: preprocessing -> scaling -> sampling -> classification
        # 'passthrough' placeholders will be replaced during hyperparameter search
        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('scaler', 'passthrough'),
            ('sampler', 'passthrough'),
            ('clf', model)
        ])
        return pipe

    def _prepare_data(self, data):
        X = data.drop(columns="target")
        y = data["target"]
        return X, y

    def get_scalers_and_samplers_grid(self):
        return {
            "scaler": [
                StandardScaler(),
                MinMaxScaler(),
                "passthrough"  # No scaling
            ],
            "sampler": [
                "passthrough",  # No resampling
                RandomOverSampler(random_state=self.random_state),
                SMOTE(random_state=self.random_state),
                RandomUnderSampler(random_state=self.random_state)
            ]
        }

    def run_pipeline(self, data, model_name, model_file, preprocessing_file, models_dir):

        # Prepare features and target
        X, y = self._prepare_data(data)

        # Create base pipeline
        pipe = self.create_pipeline(model_name, preprocessing_file)

        # Combine preprocessing grid (scalers/samplers) with model hyperparameters
        param_distributions = {
            **self.get_scalers_and_samplers_grid(),
            **self.get_param_distribution(model_file, model_name)
        }

        # Configure cross-validation strategy
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        # Perform randomized hyperparameter search
        start_time = time.time()
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions,
            n_iter=30,  # Number of random combinations to try
            scoring="roc_auc",  # Optimization metric
            cv=cv,
            n_jobs=-1,  # Use all CPU cores
            verbose=0,
        )
        search.fit(X, y)
        train_time = time.time() - start_time

        # Extract best model and parameters
        best_estimator = search.best_estimator_
        best_params = search.best_params_

        # Parse scaler and sampler names from best parameters
        scaler = best_params["scaler"]
        sampler = best_params["sampler"]

        scaler_name = (
            type(scaler).__name__ if scaler != "passthrough" else "passthrough"
        )
        sampler_name = (
            type(sampler).__name__ if sampler != "passthrough" else "passthrough"
        )

        save_machine_learning_model(
            model=best_estimator,
            directory=models_dir,
            model_name=model_name,
            scaler_name=scaler_name,
            sampler_name=sampler_name
        )
        # Save metrics and configuration to results
        result = save_params_model_with_best_params(
            model=model_name,
            scaler=scaler_name,
            balancing_name=sampler_name,
            training_time=train_time,
            cv_roc_auc=search.best_score_,  # Best cross-validation ROC AUC score
            best_params=best_params,
        )
        return [result]

    def run_all_models(self, data, model_file, preprocessing_file, models_dir):
        # Get list of all available model names
        all_model_names = self.get_model_class().keys()
        all_results = []

        # Iterate through each model and run the pipeline
        for model_name in all_model_names:
            print(f"\n{'=' * 50}")
            print(f"START PROCESSING MODEL: {model_name}")
            print(f"{'=' * 50}")

            # Train model and collect results
            results_for_model = self.run_pipeline(data, model_name, model_file, preprocessing_file, models_dir)
            all_results.extend(results_for_model)

        return all_results

    # def evaluate_model_on_valid_test(self, model_path, df, has_target=False):
    #
    #     base = os.path.splitext(os.path.basename(model_path))[0]
    #     parts = base.split("_")
    #     model_name = parts[0]
    #     scaler_name = parts[1] if len(parts) > 1 else "unknown"
    #     balancing_name = parts[2] if len(parts) > 2 else "unknown"
    #
    #     model = joblib.load(model_path)
    #
    #     X = df.drop(columns="target") if has_target else df
    #
    #     y_pred = model.predict(X)
    #     y_proba = model.predict_proba(X)[:, 1]
    #
    #     if has_target:
    #         y = df["target"]
    #         result_test_data = save_params_test_data(
    #             model=model_name,
    #             scaler=scaler_name,
    #             balancing_name=balancing_name,
    #             accuracy_score=accuracy_score(y, y_pred),
    #             precision_score=precision_score(y, y_pred),
    #             recall_score=recall_score(y, y_pred),
    #             f1_score=f1_score(y, y_pred),
    #             roc_auc_score=roc_auc_score(y, y_proba),
    #             model_path=model_path,
    #         )
    #         return [result_test_data]
    #     else:
    #         result_valid_data = save_params_valid_data(
    #             model=model_name,
    #             scaler=scaler_name,
    #             balancing_name=balancing_name,
    #             y_pred=y_pred,
    #             y_proba=y_proba,
    #             model_path=model_path,
    #         )
    #         return [result_valid_data]
