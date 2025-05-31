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
from scipy.stats import randint, uniform
warnings.filterwarnings("ignore")


df = pd.read_csv("data\\train_balanced_smote.csv")
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

param_grids = {
    "LogisticRegression": {
        "clf__C": uniform(0.01, 10),
        "clf__penalty": ['l1'],
        "clf__solver": ['liblinear', 'lbfgs']
    },
    "DecisionTreeClassifier": {
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_split": randint(2, 10)
    },
    "RandomForestClassifier": {
        "clf__n_estimators": randint(50, 200),
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_split": randint(2, 10)
    },
    "XGBClassifier": {
        "clf__n_estimators": randint(50, 200),
        "clf__max_depth": randint(3, 20),
        "clf__learning_rate": uniform(0.01, 0.3)
    },
    "lightgbm": {
        "clf__n_estimators": randint(50, 200),
        "clf__max_depth": randint(3, 20),
        "clf__learning_rate": uniform(0.01, 0.3)
    },
    "Naive Bayes": {
        
    },
    "SVC": {
        "clf__C": uniform(0.1, 10),
        "clf__kernel": ['linear', 'rbf'],
        "clf__gamma": ['scale', 'auto']
    },
    "KNeighborsClassifier": {
        "clf__n_neighbors": randint(3, 15),
        "clf__weights": ['uniform', 'distance'],
        "clf__metric": ['euclidean', 'manhattan']
    }
}

def run_random_search(pipe, param_dist, model_name, scaler_name, X, y):
    if not param_dist:
        print(f"[RandomSearch]-> Pominięto - brak parametrów do tuningu.")
        return None, None

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
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