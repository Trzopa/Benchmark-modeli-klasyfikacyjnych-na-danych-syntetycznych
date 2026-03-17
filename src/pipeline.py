import copy
import time
import warnings
from itertools import product
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import save_params_model_with_best_params, to_dataframe, prepare_data
from config.experiment_config import param_distributions

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

MODELS = {
    "LogisticRegression": LogisticRegression(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(),
    "NaiveBayes": GaussianNB(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "XGBClassifier": XGBClassifier(),
    "LGBMClassifier": LGBMClassifier(),
}

SCALERS = [
    "passthrough",  # No scaling
    StandardScaler(),
    MinMaxScaler(),
]

SAMPLERS = [
    "passthrough",  # No resampling
    RandomOverSampler(random_state=RANDOM_STATE),
    RandomUnderSampler(random_state=RANDOM_STATE),
    SMOTE(random_state=RANDOM_STATE),
]


class Benchmark:
    def __init__(self, data, preprocessing_file, save_path):
        self.data = data
        self.preprocessing_file = preprocessing_file
        self.save_path = save_path

    def __build_preprocessor(self, n_neighbors=5):
        knn_cols = []
        mean_cols = []
        drop_cols = []

        # Categorize columns by their imputation strategy
        for col, strategy in self.preprocessing_file.items():
            if strategy == 'knn':
                knn_cols.append(col)
            elif strategy == 'mean':
                mean_cols.append(col)
            elif strategy == 'delete':
                drop_cols.append(col)
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}")

        transformers = []
        if knn_cols:
            transformers.append(("knn_impute", KNNImputer(n_neighbors=n_neighbors), knn_cols))
        if mean_cols:
            transformers.append(("mean_impute", SimpleImputer(strategy="mean"), mean_cols))
        if drop_cols:
            transformers.append(("drop_cols", 'drop', drop_cols))

        return ColumnTransformer(transformers=transformers, remainder='passthrough')

    def create_pipeline(self):
        preprocessor = self.__build_preprocessor()

        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('scaler', 'passthrough'),
            ('sampler', 'passthrough'),
            ('clf', 'passthrough')
        ])
        return pipe

    # move to utils


    def run_pipeline(self, model_name, scaler, sampler):
        X, y = prepare_data()

        pipe = self.create_pipeline()
        pipe.set_params(
            clf=MODELS[model_name],
            scaler=scaler,
            sampler=sampler
        )

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions[model_name],
            n_iter=100,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        start_time = time.time()
        search.fit(X, y)
        train_time = time.time() - start_time

        result = save_params_model_with_best_params(
            model=model_name,
            scaler=scaler if scaler == "passthrough" else type(scaler).__name__,
            balancing_name=sampler if sampler == "passthrough" else type(sampler).__name__,
            training_time=train_time,
            f1=search.best_score_,
            best_params=search.best_params_
        )
        return result

    def run(self):
        all_model_names = list(MODELS.keys())
        # TODO: read about SOLID design pattern
        combinations = list(product(all_model_names, SCALERS, SAMPLERS))
        all_results = []

        for i, (model_name, scaler, sampler) in enumerate(combinations, start=1):
            print(f"Processing {i}/{len(combinations)} : {model_name} | {scaler} | {sampler}")
            result = self.run_pipeline(model_name, scaler, sampler)
            all_results.append(result)

        to_dataframe(all_results, self.save_path)


if __name__ == '__main__':
    from pathlib import Path
    from utils import load_config, load_data

    root = Path(__file__).parent
    preprocessing_file = load_config(f"{root}/config/preprocessing.yaml")
    data = load_data(f"{root}/../data/train.csv")
    bench = Benchmark(data, preprocessing_file, save_path="metrics")
    # bench.run_pipeline("LogisticRegression", "passthrough", "passthrough")
    bench.run()
