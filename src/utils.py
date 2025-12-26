import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
import json


def load_data(path):
    return pd.read_csv(path)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_all_models(models_dict, name_folder="models"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path.cwd().parent
    results_dir = root / "results" / name_folder
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"all_models_{timestamp}.pkl"
    file_path = results_dir / filename

    joblib.dump(models_dict, file_path)
    print(f"All models saved: {file_path}")
    return str(file_path)


def save_params_model_with_best_params(
        model, scaler, balancing_name, training_time, accuracy_score_val,
        precision_score_val, recall_score_val, f1_score_val, roc_auc_score_val,
        best_params=None, model_path=None
):
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
        "best_params": json.dumps(best_params) if best_params else "{}",  # ⭐ dict → string
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
