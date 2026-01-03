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


def save_params_model_with_best_params(
        model, scaler, balancing_name, training_time, accuracy_score_val,
        precision_score_val, recall_score_val, f1_score_val, roc_auc_score_val,
        best_params=None, model_path=None):
    result = {
        "model": model,
        "scaler": scaler,
        "balancing_name": balancing_name,
        "training_time": training_time,
        "accuracy_score": accuracy_score_val,
        "precision_score": precision_score_val,
        "recall_score": recall_score_val,
        "f1_score": f1_score_val,
        "roc_auc_score": roc_auc_score_val,
        "best_params": best_params,
        "model_path": model_path
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
    print(f"Wyniki zapisane do: {file_path}")
    return df


def load_models(folder):
    models = {}
    for fname in os.listdir(folder):
        if fname.endswith(".joblib"):
            path = os.path.join(folder, fname)
            name = os.path.splitext(fname)[0]
            models[name] = joblib.load(path)
    return models
