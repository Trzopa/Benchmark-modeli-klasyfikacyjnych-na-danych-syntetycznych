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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
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


    def get_balancing_methods(self):
        return {
            "SMOTE": SMOTE(random_state=self.random_state),
            "RandomOverSampler": RandomOverSampler(random_state=self.random_state),
            "RandomUnderSampler": RandomUnderSampler(random_state=self.random_state),

        }

    def create_pipeline(self, model_cls, preprocessing_file):
        pipe = ImbPipeline([
            ('preprocessor', self.build_preprocessor(preprocessing_file)),
            ('over', RandomOverSampler(random_state=self.random_state)),
            ('smote', SMOTE(random_state=self.random_state)),
            ('under', RandomUnderSampler(random_state=self.random_state)),
            ('model', model_cls)
        ])
        return pipe



    def apply_balancing(X, y, sampler):
        X_balanced, y_balanced = sampler.fit_resample(X, y)
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





def grid_search_cv(self, X, y, pipe, param_dist):
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='weighted'),
        'roc_auc': 'roc_auc',
    }

    start_time = time.time()
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=5,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
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
        'mean_test_roc_auc': results['mean_test_roc_auc'][best_index],
    }


def train_model(self, X, y, pipe, model_name, scaler_name, balancing_name):
    start_time = time.time()
    pipe.fit(X, y)
    time_training = time.time() - start_time

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
        'roc_auc': 'roc_auc',
    }

    cv_results = cross_validate(
        pipe,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
        scoring=scoring,
        n_jobs=-1
    )

    return save_params_model(
        model=model_name,
        scaler=scaler_name,
        balancing_name=balancing_name,
        training_time=time_training,
        accuracy_score_val=cv_results['test_accuracy'].mean(),
        precision_score_val=cv_results['test_precision'].mean(),
        recall_score_val=cv_results['test_recall'].mean(),
        f1_score_val=cv_results['test_f1'].mean(),
        roc_auc_score=cv_results['test_roc_auc'].mean(),
    )


def _prepare_data_and_pipelines(self, data, preprocessing_file, model_name, sampler):
    data_processed = self.preprocessing_data(data, preprocessing_file)
    X_processed = data_processed.drop(columns=["target"])
    y = data_processed["target"]

    X_balanced, y_balanced = self.apply_balancing(X_processed, y, sampler)
    model_cls = self.get_model_class(model_name)
    pipelines_to_train = {
        "StandardScaler": self.create_pipeline_with_scaler(model_cls, scaler=StandardScaler()),
        "MinMaxScaler": self.create_pipeline_with_scaler(model_cls, scaler=MinMaxScaler()),
        "None": self.create_pipeline_with_scaler(model_cls, scaler=None),
    }

    return X_balanced, y_balanced, pipelines_to_train


def run_pipeline_with_grid_search_cv(self, data, preprocessing_file, model_file, model_name):
    param_dist = self.get_param_distribution(model_file, model_name)
    all_results_list = []

    for balancing_name, sampler in self.get_balancing_methods().items():

        X_balanced, y_balanced, pipelines_to_train = self._prepare_data_and_pipelines(
            data, preprocessing_file, model_name, sampler
        )

        for scaler_name, pipe_with_scaler in pipelines_to_train.items():
            print(f"Training and tuning with {scaler_name} scaler and balancing {balancing_name}...")

            if not param_dist:
                pipe_with_scaler.fit(X_balanced, y_balanced)
                score = pipe_with_scaler.score(X_balanced, y_balanced)

                formatted_result = save_params_model_with_best_params(
                    model=model_name,
                    scaler=scaler_name,
                    balancing_name=balancing_name,
                    training_time=None,
                    accuracy_score_val=score,
                    precision_score_val=None,
                    recall_score_val=None,
                    f1_score_val=None,
                    roc_auc_score=None,
                    best_params=pipe_with_scaler.get_params()
                )
            else:
                best_results = self.grid_search_cv(X_balanced, y_balanced, pipe_with_scaler, param_dist)
                formatted_result = save_params_model_with_best_params(
                    model=model_name,
                    scaler=scaler_name,
                    balancing_name=balancing_name,
                    training_time=best_results['duration'],
                    accuracy_score_val=best_results['mean_test_accuracy'],
                    precision_score_val=best_results['mean_test_precision'],
                    recall_score_val=best_results['mean_test_recall'],
                    f1_score_val=best_results['mean_test_f1_score'],
                    roc_auc_score=best_results['mean_test_roc_auc'],
                    best_params=best_results['best_params'],
                )

            all_results_list.append(formatted_result)

    return all_results_list


def run_pipeline(self, data, preprocessing_file, model_name):
    all_results_list = []

    for balancing_name, sampler in self.get_balancing_methods().items():

        X_balanced, y_balanced, pipelines_to_train = self._prepare_data_and_pipelines(
            data, preprocessing_file, model_name, sampler
        )

        for scaler_name, pipe_with_scaler in pipelines_to_train.items():
            print(f"Training and tuning with {scaler_name} scaler and balancing {balancing_name}...")

            result = self.train_model(
                X_balanced,
                y_balanced,
                pipe_with_scaler,
                model_name,
                scaler_name,
                balancing_name
            )
            all_results_list.append(result)

    return all_results_list


def run_all_models(self, data, preprocessing_file, model_file, use_grid_search=True):
    all_model_names = self.get_model_class().keys()

    all_results = []

    for model_name in all_model_names:
        print(f"\n{'=' * 50}")
        print(f"START PROCESSING MODEL: {model_name}")
        print(f"{'=' * 50}")

        if use_grid_search:
            results_for_model = self.run_pipeline_with_grid_search_cv(data, preprocessing_file, model_file,
                                                                      model_name)
        else:

            results_for_model = self.run_pipeline(data, preprocessing_file, model_name)

        all_results.extend(results_for_model)

    return all_results


# TODO testowanie roznych parametrow, poprawa ich


if __name__ == "__main__":
