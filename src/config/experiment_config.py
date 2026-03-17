from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

RANDOM_STATE = 42

param_distributions = {

    "LogisticRegression": {
        "clf__random_state": [RANDOM_STATE],
        "clf__C": uniform(0.1, 10),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["saga", "liblinear"],
        "clf__max_iter": randint(500, 2000),
    },

    "DecisionTreeClassifier": {
        "clf__random_state": [RANDOM_STATE],
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2"],
    },

    "RandomForestClassifier": {
        "clf__random_state": [RANDOM_STATE],
        "clf__n_estimators": randint(20, 200),
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_depth": randint(2, 20),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2"],
    },

    "XGBClassifier": {
        "clf__random_state": [RANDOM_STATE],
        "clf__n_estimators": randint(50, 500),
        "clf__max_depth": randint(3, 20),
        "clf__learning_rate": uniform(0.01, 0.5),
        "clf__subsample": uniform(0.7, 0.3),
        "clf__colsample_bytree": uniform(0.7, 0.3),
        "clf__gamma": uniform(0, 5),
        "clf__reg_alpha": uniform(0, 1),
        "clf__reg_lambda": uniform(0, 1),
    },

    "LGBMClassifier": {
        "clf__verbose": [-1],
        "clf__random_state": [RANDOM_STATE],
        "clf__n_estimators": randint(100, 500),
        "clf__learning_rate": loguniform(0.01, 0.1),
        "clf__num_leaves": randint(20, 60),
        "clf__reg_alpha": loguniform(1e-8, 1.0),
        "clf__min_split_gain": uniform(0, 0.02),
    },

    "NaiveBayes": {
        # brak hiperparametrów → RandomSearch i tak zadziała
    },

    "SVC": {
        "clf__probability": [True],
        "clf__random_state": [RANDOM_STATE],
        "clf__C": uniform(0.1, 10),
        "clf__kernel": ["linear", "rbf"],
        "clf__tol": [0.01],
        "clf__cache_size": [1000],
    },

    "KNeighborsClassifier": {
        "clf__n_neighbors": randint(3, 15),
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
        "clf__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "clf__leaf_size": randint(10, 100),
    }

}
