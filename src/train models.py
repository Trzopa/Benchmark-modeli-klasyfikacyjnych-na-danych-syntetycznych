import json
import os
import warnings
from datetime import datetime
import csv
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import set_config

# Set output of transformers to be DataFrames
set_config(transform_output="pandas")
# Ignore common warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*", category=UserWarning)
warnings.filterwarnings("ignore")
# Define base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Load and prepare raw training data ===
df_raw = pd.read_csv(os.path.join(base_dir, "data", "train.csv"))

# Define imputation strategy per feature
imputation_strategies = {
    'feature_0': 'knn', 'feature_1': 'knn', 'feature_3': 'knn', 'feature_5': 'knn',
    'feature_12': 'delete', 'feature_14': 'delete', 'feature_15': 'delete',
    'feature_16': 'mean', 'feature_17': 'mean', 'feature_19': 'knn', 'feature_20': 'knn',
    'feature_21': 'delete', 'feature_24': 'knn'
}


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


# === Final cleaned dataset ===
df_clean = apply_imputations(df_raw, imputation_strategies)
X = df_clean.drop(columns=['target'])
y = df_clean['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# === Define scalers and models ===
scalers = {
    "NoScaling": None,
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler()
}

models = {
    "LogisticRegression": lambda: LogisticRegression(),
    "DecisionTreeClassifier": lambda: DecisionTreeClassifier(),
    "RandomForestClassifier": lambda: RandomForestClassifier(),
    "XGBClassifier": lambda: XGBClassifier(),
    "LightGBM": lambda: lgb.LGBMClassifier(),
    "Naive Bayes": lambda: GaussianNB(),
    "SVC": lambda: SVC(probability=True),
    "KNeighborsClassifier": lambda: KNeighborsClassifier()
}

# === Define hyperparameter search grids ===
param_grids = {
    "LogisticRegression": {
        "clf__C": uniform(0.1, 10),
        "clf__penalty": ['l2', 'l1'],
        "clf__solver": ['saga', 'liblinear'],
        "clf__max_iter": randint(500, 2000)
    },
    "DecisionTreeClassifier": {
        "clf__criterion": ['gini', 'entropy', 'log_loss'],
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ['sqrt', 'log2']
    },
    "RandomForestClassifier": {
        "clf__n_estimators": randint(20, 200),
        "clf__criterion": ['gini', 'entropy', 'log_loss'],
        "clf__max_depth": randint(2, 20),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ['sqrt', 'log2']
    },
    "XGBClassifier": {
        "clf__n_estimators": randint(50, 500),
        "clf__max_depth": randint(3, 20),
        "clf__learning_rate": uniform(0.01, 0.5),
        "clf__subsample": uniform(0.7, 0.3),
        "clf__colsample_bytree": uniform(0.7, 0.3),
        "clf__gamma": uniform(0, 5),
        "clf__reg_alpha": uniform(0, 1),
        "clf__reg_lambda": uniform(0, 1)
    },
    "LightGBM": {
        "clf__n_estimators": randint(50, 500),
        "clf__learning_rate": uniform(0.01, 0.5),
        "clf__num_leaves": randint(31, 100),
        "clf__max_depth": randint(5, 20),
        "clf__reg_alpha": uniform(0, 1),
        "clf__reg_lambda": uniform(0, 1),
        "clf__bagging_fraction": uniform(0.5, 0.5),
        "clf__feature_fraction": uniform(0.5, 0.5)

    },
    "Naive Bayes": {},
    "SVC": {
        "clf__C": uniform(0.1, 10),
        "clf__kernel": ['linear', 'rbf', 'poly', 'sigmoid'],
        "clf__gamma": ['scale', 'auto']
    },
    "KNeighborsClassifier": {
        "clf__n_neighbors": randint(3, 15),
        "clf__weights": ['uniform', 'distance'],
        "clf__metric": ['euclidean', 'manhattan'],
        "clf__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "clf__leaf_size": randint(10, 100)
    }
}

# === Scoring metrics ===
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}


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


def log_best_params(estimator_name, preprocessor_name, best_score, best_params):
    out_path = os.path.join(base_dir, "results", "metrics", "best_params.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        with open(out_path, "r") as f:
            existing = json.load(f)
    except:
        existing = {}
    key = f"{estimator_name} + {preprocessor_name}"
    existing[key] = {"best_score": round(best_score, 3), "best_params": best_params}
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=4)


# === Evaluation and pipeline logic ===
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

def evaluate_pipeline(estimator_name, preprocessor_name, preprocessor, X_train_features, y_train_labels, X_test_features, y_test_labels, is_training=True):
    if preprocessor is None and estimator_name == "SVC":
        print(f"  Skipping NoScaling for {estimator_name} (requires scaled data)")
        return
    pipe_steps = []
    if preprocessor is not None:
        pipe_steps.append(("scaler", preprocessor))

    pipe_steps.append(("smote", SMOTE(random_state=42)))

    model_factory = models[estimator_name]
    estimator = model_factory() if callable(model_factory) else model_factory
    if estimator_name == "LightGBM":
        estimator = lgb.LGBMClassifier(**estimator.get_params(), verbose=-1)

    pipe_steps.append(("clf", estimator))  # <- zawsze dodajemy klasyfikator

    pipe = ImbPipeline(pipe_steps)

    if is_training:
        metrics = cross_validate(pipe, X_train_features, y_train_labels, cv=5, scoring=scoring)
        print(f"  Scaler: {preprocessor_name:<15} | "
              f"Acc: {metrics['test_accuracy'].mean():.3f}  "
              f"Prec: {metrics['test_precision'].mean():.3f}  "
              f"Rec: {metrics['test_recall'].mean():.3f}  "
              f"F1: {metrics['test_f1'].mean():.3f}  "
              f"ROC_AUC: {metrics['test_roc_auc'].mean():.3f}  "
              f"Train Time: {metrics['fit_time'].mean():.3f}s  "
              f"Test Time: {metrics['score_time'].mean():.3f}s")
        param_dist = param_grids.get(estimator_name, {})
        if param_dist:
            run_random_search(pipe, param_dist, estimator_name, preprocessor_name, X_train_features, y_train_labels)
    else:
        pipe.fit(X_train_features, y_train_labels)
        y_pred = pipe.predict(X_test_features)
        y_proba = pipe.predict_proba(X_test_features)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
        print(f"  Scaler: {preprocessor_name:<15} | "
              f"Accuracy:  {accuracy_score(y_test_labels, y_pred):.3f}  "
              f"Precision: {precision_score(y_test_labels, y_pred):.3f}  "
              f"Recall:    {recall_score(y_test_labels, y_pred):.3f}  "
              f"F1-score:  {f1_score(y_test_labels, y_pred):.3f}", end='')
        if y_proba is not None:
            print(f"  ROC AUC: {roc_auc_score(y_test_labels, y_proba):.3f}")
        else:
            print("  ROC AUC: brak (model nie wspiera predict_proba)")


# === Run all models with all scalers ===
if __name__ == "__main__":
    print(
        "-------------------------------------  BENCHMARK  ------------------------------------------------------------")
    print(" --------- Wyniki na danych treningowych --------------")
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        for scaler_name, scaler in scalers.items():
            evaluate_pipeline(model_name, scaler_name, scaler, X_train, y_train, X_test, y_test, is_training=True)

    print(" --------- Wyniki na danych testowych ----------------")
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        for scaler_name, scaler in scalers.items():
            evaluate_pipeline(model_name, scaler_name, scaler, X_train, y_train, X_test, y_test, is_training=False)