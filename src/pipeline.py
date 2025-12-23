import inspect
import time
import warnings
from pathlib import Path

import uniform
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import save_params_model, save_params_model_with_best_params

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")


class Pipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.DIST_MAP = {"randint": randint, "uniform": uniform}

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
        return models.get(model_name)

    def create_pipeline(self, model_file, preprocessing_file, scaler='passthrough', sampler='passthrough'):
        model = self.get_model_class(model_file)

        pipe = ImbPipeline([
            ('preprocessor', self.build_preprocessor(preprocessing_file)),
            ('scaler', scaler),
            ('sampler', sampler),
            ('clf', model())
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

    def _train_one_combination(self, X, y, model_name, preprocessing_file, scaler, sampler, param_dist, sampler_name):
        pipe = self.create_pipeline(model_name, preprocessing_file, scaler=scaler, sampler=sampler)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=5,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        start_time = time.time()
        search.fit(X, y)
        training_time = time.time() - start_time

        y_pred = search.best_estimator_.predict(X)

        y_proba = search.best_estimator_.predict_proba(X)[:, 1]

        result = save_params_model(
            model=model_name,
            scaler=type(scaler).__name__ if scaler != 'passthrough' else 'passthrough',
            balancing_name=sampler_name,
            training_time=training_time,
            accuracy_score_val=accuracy_score(y, y_pred),
            precision_score_val=precision_score(y, y_pred),
            recall_score_val=recall_score(y, y_pred),
            f1_score_val=f1_score(y, y_pred),
            roc_auc_score_val=roc_auc_score(y, y_proba)
        )
        return result

    def run_pipeline_with_grid_search_cv(self, data, preprocessing_file, model_file, model_names=None):
        X = data.drop(columns="target")
        y = data["target"]
        results = []

        scalers = [StandardScaler(), MinMaxScaler(), 'passthrough']
        samplers = [
            ('none', 'passthrough'),
            ('ROS', RandomOverSampler(random_state=self.random_state)),
            ('SMOTE', SMOTE(random_state=self.random_state)),
            ('RUS', RandomUnderSampler(random_state=self.random_state))
        ]

        for model_name in model_names:
            param_dist = self.get_param_distribution(model_file, model_name)
            for scaler, (sampler_name, sampler_obj) in product(scalers, samplers):
                result = self._train_one_combination(
                    X, y, model_name, preprocessing_file, scaler, sampler_obj, param_dist, sampler_name
                )
                results.append(result)

        return results


def run_all_models(self, data, preprocessing_file, model_file):
    all_model_names = self.get_model_class().keys()

    all_results = []

    for model_name in all_model_names:
        print(f"\n{'=' * 50}")
        print(f"START PROCESSING MODEL: {model_name}")
        print(f"{'=' * 50}")

        results_for_model = self.run_pipeline_with_grid_search_cv(data, preprocessing_file, model_file,
                                                                  model_name)
        all_results.extend(results_for_model)

    return all_results

# TODO testowanie roznych parametrow, poprawa ich
