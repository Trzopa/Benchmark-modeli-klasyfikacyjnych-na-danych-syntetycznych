import json
import os
import warnings


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

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*", category=UserWarning)
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df_raw = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))

imputation_strategies = {
    'feature_0': 'knn', 'feature_1': 'knn', 'feature_3': 'knn', 'feature_5': 'knn',
    'feature_12': 'delete', 'feature_14': 'delete', 'feature_15': 'delete',
    'feature_16': 'mean', 'feature_17': 'mean', 'feature_19': 'knn', 'feature_20': 'knn',
    'feature_21': 'delete', 'feature_24': 'knn'
}


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


df_clean = apply_imputations(df_raw, imputation_strategies)
X = df_clean.drop(columns=['target'])
y = df_clean['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scalers = {
    "NoScaling": None,
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler()
}

models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier(),
    "LightGBM": lgb.LGBMClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVC": SVC(probability=True),
    "KNeighborsClassifier": KNeighborsClassifier()
}

param_grids = {
    "LogisticRegression": {
        "clf__C": uniform(0.1, 10), "clf__penalty": ['l2', 'l1'],
        "clf__solver": ['saga', 'liblinear'], "clf__max_iter": randint(500, 2000)
    },
    "DecisionTreeClassifier": {
        "clf__criterion": ['gini', 'entropy', 'log_loss'],
        "clf__max_depth": randint(3, 10),
        "clf__min_samples_split": randint(2, 10),
        "clf__min_samples_leaf": randint(1, 5),
        "clf__max_features": ['sqrt', 'log2']
    },
    "RandomForestClassifier": {
        "clf__n_estimators": randint(20, 200),
        "clf__criterion": ['gini', 'entropy', 'log_loss'],
        "clf__max_depth": randint(2, 20),
        "clf__min_samples_split": randint(2, 10),
        "clf__min_samples_leaf": randint(1, 5),
        "clf__max_features": ['sqrt', 'log2'],
        "clf__bootstrap": [True, False]
    },
    "XGBClassifier": {
        "clf__n_estimators": randint(50, 500),
        "clf__max_depth": randint(3, 20),
        "clf__learning_rate": uniform(0.01, 0.3),
        "clf__subsample": uniform(0.7, 0.29),
        "clf__colsample_bytree": uniform(0.7, 0.29),
        "clf__gamma": uniform(0, 5),
        "clf__min_child_weight": randint(1, 10),
        "clf__reg_alpha": uniform(0, 1),
        "clf__reg_lambda": uniform(0, 1)
    },
    "LightGBM": {
        "clf__n_estimators": randint(100, 400),
        "clf__learning_rate": uniform(0.01, 0.1),
        "clf__num_leaves": randint(31, 100),
        "clf__min_data_in_leaf": randint(5, 20),
        "clf__max_depth": randint(5, 20),
        "clf__feature_fraction": uniform(0.6, 0.4),
        "clf__bagging_fraction": uniform(0.6, 0.4),
        "clf__reg_alpha": uniform(0, 1),
        "clf__reg_lambda": uniform(0, 1)
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

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}


def clean_params(params, round_digits=3):
    def convert(v):
        if isinstance(v, (np.floating, float)):
            return round(float(v), round_digits)
        elif isinstance(v, (np.integer, int)):
            return int(v)
        return v

    return {k: convert(v) for k, v in params.items()}


def run_random_search(pipel, param_distinct, model_name, scaler_name, X, y):
    if not param_distinct:
        print(f"[RandomSearch] {model_name} + {scaler_name} → pominięto (brak parametrów).")
        return None, None
    print(f"[RandomSearch] {model_name} + {scaler_name} → start...")
    search = RandomizedSearchCV(
        estimator=pipel, param_distributions=param_distinct, n_iter=20,
        scoring='f1', cv=5, random_state=42, n_jobs=-1, verbose=1, error_score="raise")
    search.fit(X, y)
    best = clean_params(search.best_params_)
    print(f"   → Najlepszy F1 ({model_name} + {scaler_name}): {search.best_score_:.3f}")
    print(f"   → Parametry dla {model_name}:")
    print(json.dumps(best, indent=4))

    return search.best_score_, best


def evaluate_pipeline(model_name, model, scaler_name, scaler, X_train, y_train, X_test, y_test, is_training=True):
    if scaler is None and model_name == "SVC":
        print(f"  Skipping NoScaling for {model_name} (requires scaled data)")
        return
    pipe_steps = []
    if scaler is not None:
        pipe_steps.append(("scaler", scaler))
    pipe_steps += [
        ("smote", SMOTE(random_state=42)),
        ("clf", lgb.LGBMClassifier(verbose=-1))
    ]
    pipe = ImbPipeline(pipe_steps)
    if is_training:
        metrics = cross_validate(pipe, X_train, y_train, cv=5, scoring=scoring)
        print(f"  Scaler: {scaler_name:<15} | "
              f"Acc: {metrics['test_accuracy'].mean():.3f}  "
              f"Prec: {metrics['test_precision'].mean():.3f}  "
              f"Rec: {metrics['test_recall'].mean():.3f}  "
              f"F1: {metrics['test_f1'].mean():.3f}  "
              f"ROC_AUC: {metrics['test_roc_auc'].mean():.3f}  "
              f"Train Time: {metrics['fit_time'].mean():.3f}s  "
              f"Test Time: {metrics['score_time'].mean():.3f}s")
        param_dist = param_grids.get(model_name, {})
        if param_dist:
            run_random_search(pipe, param_dist, model_name, scaler_name, X_train, y_train)
    else:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
        print(f"  Scaler: {scaler_name:<15} | "
              f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}  "
              f"Precision: {precision_score(y_test, y_pred):.3f}  "
              f"Recall:    {recall_score(y_test, y_pred):.3f}  "
              f"F1-score:  {f1_score(y_test, y_pred):.3f}", end='')
        if y_proba is not None:
            print(f"  ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
        else:
            print("  ROC AUC: brak (model nie wspiera predict_proba)")


print("-------------------------------------  BENCHMARK  ------------------------------------------------------------")
print(" --------- Wyniki na danych treningowych --------------")
for model_name, model in models.items():
    print(f"\nModel: {model_name}")
    for scaler_name, scaler in scalers.items():
        evaluate_pipeline(model_name, model, scaler_name, scaler, X_train, y_train, X_test, y_test, is_training=True)

print(" --------- Wyniki na danych testowych ----------------")
for model_name, model in models.items():
    print(f"\nModel: {model_name}")
    for scaler_name, scaler in scalers.items():
        evaluate_pipeline(model_name, model, scaler_name, scaler, X_train, y_train, X_test, y_test, is_training=False)
