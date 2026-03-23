import ast
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42

MODELS = {
    "LogisticRegression": LogisticRegression(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(),
    "NaiveBayes": GaussianNB(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier(),
    "LGBMClassifier": LGBMClassifier(),
}

SCALERS = {
    "passthrough": "passthrough",  # No scaling
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
}

SAMPLERS = {
    "passthrough": "passthrough",  # No resampling
    "RandomOverSampler": RandomOverSampler(random_state=RANDOM_STATE),
    "RandomUnderSampler": RandomUnderSampler(random_state=RANDOM_STATE),
    "SMOTE": SMOTE(random_state=RANDOM_STATE),
}


def prepare_data(data):
    if "target" in data.columns:
        X = data.drop(columns="target")
        y = data["target"]
        return X, y
    else:
        return data, None


def __build_preprocessor(preprocessing_file, n_neighbors=5):
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


def create_pipeline(preprocessing_file):
    preprocessor = __build_preprocessor(preprocessing_file)

    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('scaler', 'passthrough'),
        ('sampler', 'passthrough'),
        ('clf', 'passthrough')
    ])
    return pipe


def load_models(folder):
    models = {}
    for file in os.listdir(folder):
        if file.endswith(".joblib"):
            path = os.path.join(folder, file)
            name = os.path.splitext(file)[0]
            models[name] = joblib.load(path)
    return models


def load_data(path):
    return pd.read_csv(path)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def evaluate_valid(y_pred, y_proba, config, training_duration):
    return save_params_model_with_evaluate_valid_data(
        model=config["model"],
        scaler=config["scaler"],
        balancing_name=config["sampler"],
        training_time=training_duration,
        predictions=y_pred.tolist(),
        y_proba=y_proba.tolist()

    )


def evaluate_test(y_test, y_pred, y_proba, config, training_duration):
    return save_params_model_with_evaluate_test_data(
        model=config["model"],
        scaler=config["scaler"],
        balancing_name=config["sampler"],
        training_time=training_duration,
        accuracy_score=accuracy_score(y_test, y_pred),
        precision_score=precision_score(y_test, y_pred),
        recall_score=recall_score(y_test, y_pred),
        f1_score=f1_score(y_test, y_pred),
        roc_auc_score=roc_auc_score(y_test, y_proba),
    )


def save_params_model_with_best_params(model, scaler, balancing_name, training_time, f1, best_params):
    result = {
        "model": model,
        "scaler": scaler,
        "balancing_name": balancing_name,
        "training_time": training_time,
        "f1": f1,
        "best_params": best_params,
    }

    return result


def save_params_model_with_evaluate_valid_data(model, scaler, balancing_name, training_time, y_proba, predictions):
    result = {
        "model": model,
        "scaler": scaler,
        "balancing_name": balancing_name,
        "training_time": training_time,
        "predictions": predictions,
        "y_proba": y_proba
    }

    return result


def save_params_model_with_evaluate_test_data(
        model, scaler, balancing_name, accuracy_score,
        precision_score, recall_score, f1_score, roc_auc_score, training_time):
    result = {
        "model": model,
        "scaler": scaler,
        "balancing_name": balancing_name,
        "training_time": training_time,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score
    }
    return result


def to_dataframe(results_list, name_folder):
    df = pd.DataFrame(results_list)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path.cwd().parent
    results_dir = root / "results" / name_folder
    results_dir.mkdir(parents=True, exist_ok=True)
    file_path = results_dir / f"results_{timestamp}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved to: {file_path}")
    return df
