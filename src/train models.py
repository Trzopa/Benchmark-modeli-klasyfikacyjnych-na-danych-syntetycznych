import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*", category=UserWarning)
warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "train_balanced_smote.csv"))

X = df.drop(columns=['target'])
y = df['target']
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
    "LightGBM": lgb.LGBMClassifier(verbose=-1),
    "Naive Bayes": GaussianNB(),
    "SVC": SVC(probability=True),
    "KNeighborsClassifier": KNeighborsClassifier()

}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

param_grids = {
    # Logistic Regression
    "LogisticRegression": {
        "clf__C": uniform(0.1, 10),  # Siła regularyzacji (niższa = mocniejsza regularyzacja)
        "clf__penalty": ['l2', 'l1'],  # Typ regularyzacji
        "clf__solver": ['saga', 'liblinear'],  # Algorytm optymalizacji
        "clf__max_iter": randint(200, 1000)  # Maksymalna liczba iteracji do zbieżności
    },

    # Decision Tree
    "DecisionTreeClassifier": {
        "clf__criterion": ['gini', 'entropy', 'log_loss'],  # Funkcja oceny podziału
        "clf__max_depth": randint(3, 10),  # Maksymalna głębokość drzewa
        "clf__min_samples_split": randint(2, 10),  # Min. liczba próbek do podziału
        "clf__min_samples_leaf": randint(1, 5),  # Min. liczba próbek w liściu
        "clf__max_features": ['sqrt', 'log2']  # Ilość cech do rozważenia przy podziale
    },

    # Random Forest
    "RandomForestClassifier": {
        "clf__n_estimators": randint(20, 200),  # Liczba drzew
        "clf__criterion": ['gini', 'entropy', 'log_loss'],  # Funkcja podziału
        "clf__max_depth": randint(2, 20),  # Maksymalna głębokość
        "clf__min_samples_split": randint(2, 10),  # Minimalna liczba próbek do podziału
        "clf__min_samples_leaf": randint(1, 5),  # Minimalna liczba próbek w liściu
        "clf__max_features": ['sqrt', 'log2'],  # Liczba cech do rozważenia
        "clf__bootstrap": [True, False],  # Czy stosować bootstrapping

    },

    # XGBoost
    "XGBClassifier": {
        "clf__n_estimators": randint(50, 500),  # Liczba drzew
        "clf__max_depth": randint(3, 20),  # Maksymalna głębokość
        "clf__learning_rate": uniform(0.01, 0.3),  # Szybkość uczenia
        "clf__subsample": uniform(0.7, 0.29),  # Część próbek do trenowania
        "clf__colsample_bytree": uniform(0.7, 0.29),  # Część cech do budowy drzewa
        "clf__gamma": uniform(0, 5),  # Minimalna redukcja strat
        "clf__min_child_weight": randint(1, 10),  # Minimalna liczba próbek w liściu
        "clf__reg_alpha": uniform(0, 1),  # L1 regularization
        "clf__reg_lambda": uniform(0, 1)  # L2 regularization
    },

    # LightGBM
    "LightGBM": {
    "clf__n_estimators": randint(50, 400),
    "clf__max_depth": randint(3, 15),
    "clf__learning_rate": uniform(0.05, 0.25),
    "clf__bagging_fraction": uniform(0.6, 0.35),        # zamiast subsample
    "clf__feature_fraction": uniform(0.6, 0.35),        # zamiast colsample_bytree
    "clf__min_split_gain": uniform(0, 2),               # minimalna redukcja strat (jak gamma)
    "clf__min_child_weight": randint(1, 10),
    "clf__reg_alpha": uniform(0, 1),
    "clf__reg_lambda": uniform(0, 1)
    },


    # Naive Bayes (brak parametrów do strojenia)
    "Naive Bayes": {},

    # Support Vector Classifier
    "SVC": {
        "clf__C": uniform(0.1, 10),  # Karność za błędy klasyfikacji
        "clf__kernel": ['linear', 'rbf', 'poly', 'sigmoid'],  # Typ jądra
        "clf__gamma": ['scale', 'auto'],  # Parametr dla jądra RBF
    },

    # K-Nearest Neighbors
    "KNeighborsClassifier": {
        "clf__n_neighbors": randint(3, 15),  # Liczba sąsiadów
        "clf__weights": ['uniform', 'distance'],  # Wagi dla sąsiadów
        "clf__metric": ['euclidean', 'manhattan'],  # Miara odległości
        "clf__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorytm wyszukiwania
        "clf__leaf_size": randint(10, 100)  # Rozmiar liścia (dot. alg. drzewiastych)
    }
}


def clean_params(params, round_digits=3):
    def convert(v):
        if isinstance(v, (np.floating, float)):
            return round(float(v), round_digits)
        elif isinstance(v, (np.integer, int)):
            return int(v)
        return v

    return {k: convert(v) for k, v in params.items()}


def run_random_search(pipe, param_dist, model_name, scaler_name, X, y):
    if not param_dist:
        print("[RandomSearch] -> pominięto (brak parametrów).")
        return None, None

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        error_score="raise"  # od razu zobaczysz nielegalne kombinacje
    )
    search.fit(X, y)

    best = clean_params(search.best_params_)
    print(f"   → Najlepszy F1: {search.best_score_:.3f}")
    print(f"   → Parametry:\n{json.dumps(best, indent=4)}")
    return search.best_score_, best

results = []

print("-------------------------------------  BENCHMARK  ------------------------------------------------------------")

for model_name, model in models.items():
    print(f"\nModel: {model_name}")

    for scaler_name, scaler in scalers.items():
        if scaler is None and model_name in ["SVC"]:
            print(f"  Skipping NoScaling for {model_name} (requires scaled data)")
            continue

        pipe = Pipeline([("clf", model)]) if scaler is None else Pipeline([("scaler", scaler), ("clf", model)])

        metrics = cross_validate(pipe, X, y, scoring=scoring, cv=5)

        print(f"  Scaler: {scaler_name:<15} | "
              f"Acc: {metrics['test_accuracy'].mean():.3f}  "
              f"Prec: {metrics['test_precision'].mean():.3f}  "
              f"Rec: {metrics['test_recall'].mean():.3f}  "
              f"F1: {metrics['test_f1'].mean():.3f}  "
              f"ROC_AUC: {metrics['test_roc_auc'].mean():.3f}  "
              f"Train Time: {metrics['fit_time'].mean():.3f}s  "
              f"Test Time: {metrics['score_time'].mean():.3f}s")

        param_dist = param_grids.get(model_name, {})
        best_score, best_params = None, None
        if param_dist:
            best_score, best_params = run_random_search(pipe, param_dist, model_name, scaler_name, X_train, y_train)

        results.append({
            "model": model_name,
            "scaler": scaler_name,
            "accuracy": round(metrics['test_accuracy'].mean(), 3),
            "precision": round(metrics['test_precision'].mean(), 3),
            "recall": round(metrics['test_recall'].mean(), 3),
            "f1": round(metrics['test_f1'].mean(), 3),
            "roc_auc": round(metrics['test_roc_auc'].mean(), 3),
            "train_time": round(metrics['fit_time'].mean(), 3),
            "test_time": round(metrics['score_time'].mean(), 3),
            "random_search_score": round(best_score, 3) if best_score is not None else None,
            "best_params": json.dumps(best_params) if best_params is not None else None
        })

results_df = pd.DataFrame(results)
output_path = os.path.join(BASE_DIR, "benchmark_results.csv")
results_df.to_csv(output_path, index=False)
print(f"Wyniki zapisane do: {output_path}")