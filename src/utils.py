import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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

SCALERS = [
    "passthrough",  # No scaling
    StandardScaler(),
    MinMaxScaler(),
]

SAMPLERS = [
    "passthrough",  # No resampling
    RandomOverSampler(random_state=RANDOM_STATE),
    RandomUnderSampler(random_state=RANDOM_STATE),
    SMOTE(random_state=RANDOM_STATE),
]


def load_data(path):
    return pd.read_csv(path)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(self):
    if "target" in self.data.columns:
        X = self.data.drop(columns="target")
        y = self.data["target"]
        return X, y
    else:
        return self.data, None


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


def load_models(folder):
    models = {}
    for file in os.listdir(folder):
        if file.endswith(".joblib"):
            path = os.path.join(folder, file)
            name = os.path.splitext(file)[0]
            models[name] = joblib.load(path)
    return models


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
