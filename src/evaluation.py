import ast
import time

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import save_params_model_with_evaluate_valid_data, \
    save_params_model_with_evaluate_test_data
from pipeline import BenchmarkPipeline


class ModelEvaluator(BenchmarkPipeline):
    def __init__(self):
        super().__init__()

    def __prepare_data(self, data):
        if "target" in data.columns:
            X = data.drop(columns="target")
            y = data["target"]
            return X, y
        else:
            return data, None

    def __get_scaler_from_name(self, name):
        mapping = {
            "passthrough": "passthrough",
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
        }
        return mapping[name]

    def __get_sampler_from_name(self, name):
        mapping = {
            "passthrough": "passthrough",
            "RandomOverSampler": RandomOverSampler(random_state=self.random_state),
            "RandomUnderSampler": RandomUnderSampler(random_state=self.random_state),
            "SMOTE": SMOTE(random_state=self.random_state),
        }
        return mapping[name]

    def get_configs(self, results_df):
        all_configs = []
        for _, row in results_df.iterrows():
            params = self.__parse_best_params(row['best_params'])

            configs = {
                'model': row['model'],
                'scaler': row['scaler'],
                'sampler': row['balancing_name'],
                'params': params
            }
            all_configs.append(configs)
        return all_configs

    def __parse_best_params(self, params_str):
        clean_str = params_str.replace('np.float64(', '').replace(')', '')
        return ast.literal_eval(clean_str)

    def __train_and_predict(self, config, X_train, y_train, X_eval, preprocessing_file):
        scaler = self.__get_scaler_from_name(config["scaler"])
        sampler = self.__get_sampler_from_name(config["sampler"])

        pipe = self.create_pipeline(config["model"], preprocessing_file)
        pipe.set_params(scaler=scaler, sampler=sampler, **config["params"])

        start_time = time.time()
        pipe.fit(X_train, y_train)
        training_duration = time.time() - start_time

        y_pred = pipe.predict(X_eval)
        y_proba = pipe.predict_proba(X_eval)[:, 1]

        return y_pred, y_proba, training_duration

    def __evaluate_valid(self, y_pred, y_proba, config, training_duration):
        return save_params_model_with_evaluate_valid_data(
            model=config["model"],
            scaler=config["scaler"],
            balancing_name=config["sampler"],
            training_time=training_duration,
            cv_roc_auc=y_proba.mean(),
            predictions=y_pred
        )

    def __evaluate_test(self, y_test, y_pred, y_proba, config, training_duration):
        return save_params_model_with_evaluate_test_data(
            model=config["model"],
            scaler=config["scaler"],
            balancing_name=config["sampler"],
            training_time=training_duration,
            accuracy_score=accuracy_score(y_test, y_pred),
            precision_score=precision_score(y_test, y_pred),
            recall_score=recall_score(y_test, y_pred),
            f1_score=f1_score(y_test, y_pred),
            roc_auc_score=roc_auc_score(y_test, y_proba),
        )

    def evaluate_to_valid_data(self, train_data, valid_data, results_df, preprocessing_file):
        configs = self.get_configs(results_df)
        X_train, y_train = self.__prepare_data(train_data)
        X_valid, _ = self.__prepare_data(valid_data)
        results = []
        for config in configs:
            y_pred, y_proba, duration = self.__train_and_predict(config, X_train, y_train, X_valid, preprocessing_file)
            result = self.__evaluate_valid(y_pred, y_proba, config, duration)
            results.append(result)
        return results

    def evaluate_to_test_data(self, train_data, test_data, results_df, preprocessing_file):
        configs = self.get_configs(results_df)
        X_train, y_train = self.__prepare_data(train_data)
        X_test, y_test = self.__prepare_data(test_data)
        results = []
        for config in configs:
            y_pred, y_proba, duration = self.__train_and_predict(config, X_train, y_train, X_test, preprocessing_file)
            result = self.__evaluate_test(y_test, y_pred, y_proba, config, duration)
            results.append(result)

        return results
