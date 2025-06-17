import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import RandomizedSearchCV

# definiujemy base_dir tylko, jeśli chcemy mieć tu wygodną ścieżkę
base_dir = Path(__file__).resolve().parent.parent


def load_config(path: str) -> Dict[str, Any]:
    ext = Path(path).suffix.lower()
    with open(path, 'r', encoding='utf-8') as f:
        if ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_metrics(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_predictions(ids: pd.Series, preds: pd.Series, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = pd.DataFrame({'id': ids, 'y_pred': preds})
    out.to_csv(path, index=False)


# === Imputation helper functions ===
def get_columns_to_delete(strategies_dict):
    return [col for col, strategy in strategies_dict.items() if strategy == 'delete']


def remove_columns(df, columns):
    return df.drop(columns=columns, errors='ignore')


def apply_knn_imputation(df, n_neighbors=5):
    knn_inputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = knn_inputer.fit_transform(df)
    return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


def apply_simple_imputation(df, strategies_dict):
    df_copy = df.copy()
    for column, strategy in strategies_dict.items():
        if strategy in ['delete', 'knn']:
            continue
        if column in df_copy.columns:
            inputer = SimpleImputer(strategy=strategy)
            df_copy[[column]] = inputer.fit_transform(df_copy[[column]])
    return df_copy


def apply_imputations(df, strategies_dict, knn_n_neighbors=5):
    df_filled = df.copy()
    df_filled = remove_columns(df_filled, get_columns_to_delete(strategies_dict))
    if 'knn' in strategies_dict.values():
        df_filled = apply_knn_imputation(df_filled, knn_n_neighbors)
    return apply_simple_imputation(df_filled, strategies_dict)


# === Utility functions ===
def clean_params(params, round_digits=3):
    def convert(v):
        if isinstance(v, (np.floating, float)):
            return round(float(v), round_digits)
        elif isinstance(v, (np.integer, int)):
            return int(v)
        return v

    return {k: convert(v) for k, v in params.items()}


def log_cv_results(estimator_name, preprocessor_name, metrics):
    out_path = os.path.join(base_dir, "results", "metrics", "train_cv_metrics.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result_row = {
        "timestamp": datetime.now().isoformat(),
        "model": estimator_name,
        "scaler": preprocessor_name,
        "accuracy": metrics["test_accuracy"].mean(),
        "precision": metrics["test_precision"].mean(),
        "recall": metrics["test_recall"].mean(),
        "f1": metrics["test_f1"].mean(),
        "roc_auc": metrics["test_roc_auc"].mean(),
        "fit_time": metrics["fit_time"].mean(),
        "score_time": metrics["score_time"].mean()
    }
    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)


def log_best_params(estimator_name: str, preprocessor_name: str, best_score: float, best_params: Dict[str, Any],
                    metric_name: str = "f1", base_dir: str = ".", random_state: int | None = None,
                    ) -> None:
    out_path = Path(base_dir) / "results" / "metrics" / "best_params_flat.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "estimator": estimator_name,
        "preprocessor": preprocessor_name,
        "metric": metric_name,
        "best_score": round(best_score, 3),
        "run_datetime": datetime.now().isoformat(timespec="seconds"),
    }
    if random_state is not None:
        row["random_state"] = random_state
    for k, v in best_params.items():
        clean_name = k.split("__")[-1]  # usuń np. 'clf__'
        row[f"param_{clean_name}"] = v

    fieldnames = list(row.keys())

    mode = "a" if out_path.is_file() else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

        if mode == "w":
            writer.writeheader()
        else:
            missing_cols = set(fieldnames) - set(writer.fieldnames)
            if missing_cols:
                raise RuntimeError(
                    f"Plik {out_path} istnieje, ale brak kolumn {missing_cols}. "
                    "Najłatwiej usuń plik lub zapisz do nowej nazwy."
                )

        writer.writerow(row)


def run_random_search(pipeline, param_grid, estimator_name, preprocessor_name, X_data, y_labels):
    if not param_grid:
        print(f"[RandomSearch] {estimator_name} + {preprocessor_name} → skipped (no parameters).")
        return None, None
    print(f"[RandomSearch] {estimator_name} + {preprocessor_name} → start...")
    search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=10, scoring='f1',
                                cv=5, random_state=42, n_jobs=-1, verbose=1, error_score="raise")
    search.fit(X_data, y_labels)
    best_params = clean_params(search.best_params_)
    print(f"   → Best F1: {search.best_score_:.3f}\n", json.dumps(best_params, indent=4))
    log_best_params(estimator_name, preprocessor_name, search.best_score_, best_params)
    return search.best_score_, best_params
