import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml


def load_data(path):
    return pd.read_csv(path)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_params_model_with_best_params(model, scaler, balancing_name, training_time, cv_roc_auc, best_params):
    result = {
        "model": model,
        "scaler": scaler,
        "balancing_name": balancing_name,
        "training_time": training_time,
        "cv_roc_auc": cv_roc_auc,
        "best_params": best_params,
    }

    return result


def save_params_model_with_evaluate_valid_data(model, scaler, balancing_name, training_time, cv_roc_auc, predictions):
    result = {
        "model": model,
        "scaler": scaler,
        "balancing_name": balancing_name,
        "training_time": training_time,
        "cv_roc_auc": cv_roc_auc,
        "predictions": predictions
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
