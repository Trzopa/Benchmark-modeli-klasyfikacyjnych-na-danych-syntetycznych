from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from imblearn.over_sampling import SMOTE
from importlib import import_module

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB


class Pipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def preprocessing_data(self, data, preprocessing_config):
        mean_cols = [col for col, method in preprocessing_config.items() if method == "mean"]
        knn_cols = [col for col, method in preprocessing_config.items() if method == "knn"]
        cols_to_delete = [col for col, method in preprocessing_config.items() if method == "delete"]
        data = data.drop(columns=cols_to_delete)
        if mean_cols:
            mean_inputer = SimpleImputer(strategy="mean")
            data[mean_cols] = mean_inputer.fit_transform(data[mean_cols])
        if knn_cols:
            knn_inputer = KNNImputer(n_neighbors=5)
            data[knn_cols] = knn_inputer.fit_transform(data[knn_cols])
        return data

    def prepare_features(self, data, target_column, preprocessing_config):
        X = data.drop(column=[target_column])
        y = data[target_column]
        X = self.preprocessing_data(X, preprocessing_config)
        return X, y

    def scaling(self, scalers_config, scaler_name):
        if scaler_name == "NoScaling" or scalers_config.get(scaler_name) is None:
            return FunctionTransformer(lambda x: x, validate=False)

        class_path = scalers_config[scaler_name]
        module_name, class_name = class_path.rsplit('.', 1)

        module = import_module(module_name)
        scaler_class = getattr(module, class_name)

        return scaler_class()

    def parse_models_config(self, models_config):
        models = []

        for model_name, params in models_config.items():
            fixed_params = {}
            search_space = {}

            for param_name, param_value in params.items():
                if isinstance(param_value, dict) and 'distribution' in param_value:
                    search_space[param_name] = param_value
                elif isinstance(param_value, list):
                    search_space[param_name] = param_value
                else:
                    fixed_params[param_name] = param_value

            models.append({
                'model_name': model_name,
                'fixed_params': fixed_params,
                'search_space': search_space
            })

        return models

    def grid_search_cv(self, pipe, random_state):
        search = RandomizedSearchCV(
            estimator=pipe,
            n_jobs=-1,
            scoring="f1",
            verbose=1)

        return search

    def run(self, data, target_column, preprocessing_config):
        X = data.drop[column=target_column]
        y = data[target_column]
        X = self.preprocessing_data(X, preprocessing_config)
        scaler1 = StandardScaler()
        scaler2 = MinMaxScaler()
        X_stan, y_stan = scaler1.fit_transform(X, y)

        X_mm, y_mm = scaler2.fit_transform(X, y)
        balance = SMOTE()
        X_b, y_b = balance.fit_resample(X, y)
        logic_cl = LogisticRegression()
        logic_cl.fit(X_b, y_b)
        treeml = DecisionTreeClassifier()
        treeml.fit(X_b, y_b)
        forestml = RandomForestClassifier()
        forestml.fit(X_b, y_b)
        xbml = XGBClassifier()
        xbml.fit(X_b, y_b)
        lxbml = LGBMClassifier()
        lxbml.fit(X_b, y_b)
        gml = GaussianNB()
        gml.fit(X_b, y_b)
        svcml = SVC()
        svcml.fit(X_b, y_b)
        knml = KNeighborsClassifier()
        knml.fit(X_b, y_b)
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)