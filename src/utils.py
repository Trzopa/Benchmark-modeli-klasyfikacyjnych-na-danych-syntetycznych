import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
import json

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_data(path):
    return pd.read_csv(path)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Model saved: {model_path}")
    return model_path


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
        "model_path": model_path or ""
    }
    return result


def predict_valid(valid_path, model_path, output_path="results/predictions/valid_predictions.csv"):
    model = joblib.load(model_path)
    valid_df = pd.read_csv(valid_path)

    y_pred = model.predict(valid_df)
    y_proba = model.predict_proba(valid_df)[:, 1]

    valid_df['y_pred'] = y_pred
    valid_df['y_proba'] = y_proba
    valid_df.to_csv(output_path, index=False)
    print(f"Valid zapisane: {output_path}")
    return valid_df


def evaluate_test(test_path, model_path, output_path="results/predictions/test_predictions.csv"):
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns="target")
    y_test = test_df["target"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    test_df['y_pred'] = y_pred
    test_df['y_proba'] = y_proba
    test_df.to_csv(output_path, index=False)
    print(f"Test metryki: {metrics}")
    return test_df, metrics


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
