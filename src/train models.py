import argparse
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import uniform, randint
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.utils import (
    load_config,
    apply_imputations,
    run_random_search,
    log_cv_results
)

model_factories = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
    "LightGBM": lambda: lgb.LGBMClassifier(verbose=-1),
    "NaiveBayes": GaussianNB,
    "SVC": lambda: SVC(probability=True),
    "KNeighborsClassifier": KNeighborsClassifier,
}

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")


def build_param_distributions(specifications: dict) -> dict:
    dist_map = {'uniform': uniform, 'randint': randint}
    pdists = {}
    for param, val in specifications.items():
        if isinstance(val, dict) and 'distribution' in val:
            dist_name = val['distribution']
            kwargs = {k: v for k, v in val.items() if k != 'distribution'}
            pdists[param] = dist_map[dist_name](**kwargs)
        else:
            pdists[param] = val
    return pdists


def evaluate_pipeline(
        name: str,
        factory,
        scaler,
        X: pd.DataFrame,
        y: pd.Series,
        raw_param_grid: dict,
        scoring: dict,
        cv: int,
        random_state: int,
        n_iter_search: int,
        n_jobs: int,
        scoring_for_search: str,
        all_params: list,
        imputation_strategies: dict
):
    if scaler is None and name == "SVC":
        print(f"  Skipping NoScaling for {name}")
        return
    imputers = []
    for col, strategy in imputation_strategies.items():
        if strategy == 'drop':
            continue
        imputers.append((f'imputer_{col}', SimpleImputer(strategy=strategy), [col]))

    preprocessor = ColumnTransformer(
        transformers=imputers,
        remainder='passthrough'
    )



    steps = [('imputer', preprocessor)]

    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("smote", SMOTE(random_state=random_state)))
    model = factory()
    steps.append(("clf", model))

    pipe = ImbPipeline(steps)
    cv_res = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=False
    )

    log_cv_results(
        estimator_name=name,
        preprocessor_name=(scaler.__class__.__name__ if scaler else "NoScaling"),
        metrics=cv_res
    )
    if raw_param_grid:
        param_dist = build_param_distributions(raw_param_grid)
        run_random_search(
            pipeline=pipe,
            param_grid=param_dist,
            estimator_name=name,
            preprocessor_name=(scaler.__class__.__name__ if scaler else "NoScaling"),
            X_data=X,
            y_labels=y,
            all_params=all_params
        )

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Benchmark modeli klasyfikacyjnych na danych syntetycznych"
    )
    parser.add_argument(
        "--config-dir",
        default=str(project_root / "src" / "config"),
        help="katalog z preprocessing.yaml, models.yaml, training.yaml, scalers.yaml, scoring.yaml"
    )
    parser.add_argument(
        "--data-dir",
        default=str(project_root / "data"),
        help="katalog z train.csv, valid.csv, test.csv"
    )
    args = parser.parse_args()

    # 1) Load configs
    cfg_pre = load_config(os.path.join(args.config_dir, "preprocessing.yaml"))
    raw_param_grids = load_config(os.path.join(args.config_dir, "models.yaml"))
    cfg_train = load_config(os.path.join(args.config_dir, "training.yaml"))
    cfg_scalers = load_config(os.path.join(args.config_dir, "scalers.yaml"))
    cfg_scoring = load_config(os.path.join(args.config_dir, "scoring.yaml"))

    all_params = set()

    for model_name, grid in raw_param_grids.items():
        for param_name in grid.keys():
            clean_name = param_name.split("__")[-1]
            all_params.add(f"param_{clean_name}")

    all_params = sorted(all_params)

    # 2) Load & clean data
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_clean = apply_imputations(df, cfg_pre["imputation_strategies"])
    target_col = cfg_pre.get("target_col", "target")
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg_train["test_size"],
        random_state=cfg_train["random_state"]
    )

    # 4) Prepare scalers
    scalers = {}
    for name, spec in cfg_scalers.items():
        if spec is None:
            scalers[name] = None
        else:
            module = __import__(spec["module"], fromlist=[spec["class"]])
            cls = getattr(module, spec["class"])
            scalers[name] = cls(**spec.get("params", {}))

    # 5) Run benchmark
    print("\n=== START BENCHMARK ===")
    for model_name, factory in model_factories.items():
        print(f"\n-- Model: {model_name} --")
        grid_spec = raw_param_grids.get(model_name, {})
        for scaler_name, scaler in scalers.items():
            print(f" Scaler: {scaler_name}")
            evaluate_pipeline(
                name=model_name,
                factory=factory,
                scaler=scaler,
                X=X_train,
                y=y_train,
                raw_param_grid=grid_spec,
                scoring=cfg_scoring,
                cv=cfg_train["cv"],
                random_state=cfg_train["random_state"],
                n_iter_search=cfg_train.get("n_iter_search", 10),
                n_jobs=cfg_train.get("n_jobs", -1),
                scoring_for_search=cfg_train.get("scoring_for_search", "f1"),
                all_params=all_params,
                imputation_strategies=cfg_pre["imputation_strategies"]
            )

    print("\n=== BENCHMARK ZAKOŃCZONY ===")
