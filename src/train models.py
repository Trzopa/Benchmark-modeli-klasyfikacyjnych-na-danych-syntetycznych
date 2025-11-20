from sklearn.preprocessing import StandardScaler, MinMaxScaler

from imblearn.over_sampling import SMOTE
from importlib import import_module

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV


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

    def build_pipline(self):
        return Pipeline([
            ("balacing", SMOTE()),
        ])

    def prepare_features(self, data, target_column, preprocessing_config):
        X = data.drop(column=[target_column])
        y = data[target_column]
        X = self.preprocessing_data(X, preprocessing_config)
        return X, y

    def search_cv(self, pipe, random_state):
        search = GridSearchCV(
            estimator=pipe,
            n_jobs=-1,
            scoring="f1",
            verbose=1       )

        return search
