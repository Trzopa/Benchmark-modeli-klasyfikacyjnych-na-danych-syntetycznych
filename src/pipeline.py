import time
import warnings
from itertools import product
from pprint import pprint

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
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

from utils import save_params_model_with_best_params, to_dataframe

set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

DIST_PARAM = {
    "randint": randint,
    "uniform": uniform,
    "loguniform": loguniform,
}

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
    def __init__(self, data, model_file, preprocessing_file, save_path):
        self.data = data
        self.model_file = model_file
        self.preprocessing_file = preprocessing_file
        self.save_path = save_path

    def __get_param_distribution(self, model_name):
        model_config = self.model_file[model_name]
        param_distributions = {}

        for param_name, spec in model_config.items():
            if isinstance(spec, list):
                param_distributions[param_name] = spec

            elif isinstance(spec, dict) and "value" in spec:
                param_distributions[param_name] = [spec["value"]]

            elif isinstance(spec, dict) and "distribution" in spec:
                dist_name = spec["distribution"]
                dist_cls = DIST_PARAM[dist_name]

                if dist_name == "randint":
                    param_distributions[param_name] = dist_cls(low=spec["low"], high=spec["high"])
                elif dist_name == "uniform":
                    param_distributions[param_name] = dist_cls(loc=spec["loc"], scale=spec["scale"])
                elif dist_name == "loguniform":
                    param_distributions[param_name] = dist_cls(spec["low"], spec["high"])
        # TODO:
        #  model = LogisticRegression() - na przykład
        #  gdzieć tutaj dodac  param_distributions['clf'] = model
        return param_distributions

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

    def create_pipeline(self, model_name):
        model = MODELS[model_name]
        preprocessor = self.__build_preprocessor()

        # Pipeline steps: preprocessing -> scaling -> sampling -> classification
        # 'passthrough' placeholders will be replaced during hyperparameter search
        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('scaler', 'passthrough'),
            ('sampler', 'passthrough'),
            ('clf', model) # <--- TODO: dać passthrough i podać wiele modeli zamiast jednego (zamiast model podaj MODELS)
        ])
        return pipe

    # move to utils
    def prepare_data(self):
        if "target" in self.data.columns:
            X = self.data.drop(columns="target")
            y = self.data["target"]
            return X, y
        else:
            return self.data, None

    def run_pipeline(self, model_name, scaler_obj, sampler_obj):
        # Prepare features and target
        X, y = self.prepare_data()

        pipe = self.create_pipeline(model_name)

        pipe.set_params(scaler=scaler_obj, sampler=sampler_obj)
        param_distributions = self.__get_param_distribution(model_name)
        # TODO: test and remove this
        print(100*'*')
        pprint(param_distributions)
        print(100*'*')
        return None
        scaler_name = type(scaler_obj).__name__ if scaler_obj != "passthrough" else "passthrough"
        sampler_name = type(sampler_obj).__name__ if sampler_obj != "passthrough" else "passthrough"

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions, # TODO: może tutaj trzeba podać??
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
            scaler=scaler_name,
            balancing_name=sampler_name,
            training_time=train_time,
            f1=search.best_score_,
            best_params=search.best_params_,
        )
        return [result]

    def run(self):
        all_model_names = list(MODELS.keys())
        # TODO: read about SOLID design pattern

        combinations = list(product(all_model_names, SCALERS, SAMPLERS))

        all_results = []
        for model_name, scaler_obj, sampler_obj in combinations: # pozbyć się tej pę
            print(f"Preprocessing {len(all_results) + 1}/96: {model_name}")

            result = self.run_pipeline(
                model_name, scaler_obj, sampler_obj
            )
            all_results.extend(result)

        to_dataframe(all_results, self.save_path)

