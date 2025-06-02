import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import warnings
import os
from scipy.stats import randint, uniform
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "train_balanced_smote.csv"))

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scalers = {
"NoScaling":None,
"StandardScaler": StandardScaler(),
"MinMaxScaler": MinMaxScaler()
}

models ={
   "LogisticRegression" : LogisticRegression(),
   "DecisionTreeClassifier": DecisionTreeClassifier(),
   "RandomForestClassifier": RandomForestClassifier(),
   "XGBClassifier": XGBClassifier(),
   "lightgbm": lgb.LGBMClassifier(verbose=-1),
   "Naive Bayes": GaussianNB(),
   "SVC" : SVC(probability=True),
   "KNeighborsClassifier": KNeighborsClassifier()

}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

from scipy.stats import randint, uniform

param_grids = {
    "LogisticRegression": {
    "clf__C": uniform(0.1, 10),
    "clf__penalty": ['l2'],
    "clf__solver": ['lbfgs'],
    "clf__max_iter": randint(200, 1000)
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
        "clf__bootstrap": [True, False],
        "clf__oob_score": [True, False]
    },
    "XGBClassifier": {
        "clf__n_estimators": randint(50, 500),
        "clf__max_depth": randint(3, 20),
        "clf__learning_rate": uniform(0.01, 0.3),
        "clf__subsample": uniform(0.5, 0.5),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__gamma": uniform(0, 5),
        "clf__min_child_weight": randint(1, 10),
        "clf__reg_alpha": uniform(0, 1),
        "clf__reg_lambda": uniform(0, 1)
    },
    
    "lightgbm": {
    "clf__n_estimators": randint(50, 100),                # mniej drzew = szybszy trening
    "clf__learning_rate": uniform(0.05, 0.1),              # bardziej realistyczne zakresy
    "clf__max_depth": randint(3, 10),                      # ograniczamy złożoność
    "clf__num_leaves": randint(20, 80),                    # ograniczamy liczbę liści
    "clf__min_child_samples": randint(20, 100),            # większa wartość = mniej przeuczenia
    "clf__subsample": uniform(0.7, 0.3),                    # losowe próbkowanie – stabilniejsze
    "clf__colsample_bytree": uniform(0.7, 0.3),            # zmniejszamy ilość cech na drzewo
    "clf__boosting_type": ['gbdt', 'dart'],                # usuwamy 'goss' (czasem niestabilny)
    "clf__reg_alpha": uniform(0.0, 0.5),                   # mniejszy zakres regularyzacji
    "clf__reg_lambda": uniform(0.0, 0.5)
    },


    "Naive Bayes": {
        # brak parametrów do strojenia dla GaussianNB
    },
    "SVC": {
        "clf__C": uniform(0.1, 10),
        "clf__kernel": ['linear', 'rbf', 'poly', 'sigmoid'],
        "clf__gamma": ['scale', 'auto'],
        "clf__max_iter": randint(100, 1000)
    },
    "KNeighborsClassifier": {
        "clf__n_neighbors": randint(3, 15),
        "clf__weights": ['uniform', 'distance'],
        "clf__metric": ['euclidean', 'manhattan'],
        "clf__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "clf__leaf_size": randint(10, 100)
    }
}


def run_random_search(pipe, param_dist, model_name, scaler_name, X, y):
    if not param_dist:
        print(f"[RandomSearch]-> Pominięto - brak parametrów do tuningu.")
        return None, None

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=10,
        scoring='f1',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)

    print(f"Najlepszy F1: {search.best_score_:.3f}")
    print(f"Najlepsze parametry: {search.best_params_}\n")
    return search.best_score_, search.best_params_



print("-------------------------------------  BENCHMARK  ------------------------------------------------------------")

for model_name, model in models.items():
    print(f"Model: {model_name}")
    for scaler_name, scaler in scalers.items():
        if scaler is None:
            pipe = Pipeline([("clf", model)])
        else:
            pipe = Pipeline([("scaler", scaler), ("clf", model)])
        
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
        best_score, best_params = run_random_search(pipe, param_dist, model_name, scaler_name, X_train, y_train)

    print()