import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer, KNNImputer
import yaml
from sklearn.preprocessing import FunctionTransformer
from importlib import import_module


def load_data(path):
    with open(path, "r") as f:
        file = pd.read_csv(f)
    return file


def load_config(path):
    with open(path, "r") as f:
        file = yaml.safe_load(f)
    return file


def preprocessing_data(data, preprocessing_config):
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


def scaling(scalers_config, scaler_name):
    if scaler_name == "NoScaling" or scalers_config.get(scaler_name) is None:
        return FunctionTransformer(lambda x: x, validate=False)

    class_path = scalers_config[scaler_name]
    module_name, class_name = class_path.rsplit('.', 1)

    module = import_module(module_name)
    scaler_class = getattr(module, class_name)

    return scaler_class()


def parse_models_config(models_config):
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




if __name__ == "__main__":
    root = Path.cwd().parent
    # d = load_data(f"{root}/data/test.csv")
    # print(d)
    c = load_config(f"config/training.yaml")
    print(c)
