import ast
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import save_params_model_with_evaluate_valid_data, save_params_model_with_evaluate_test_data, prepare_data, \
    MODELS

RANDOM_STATE = 42


def __parse_best_params(params_str):
    clean_str = params_str.replace('np.float64(', '').replace(')', '')
    return ast.literal_eval(clean_str)


def get_configs(results_df):
    all_configs = []
    for _, row in results_df.iterrows():
        params = __parse_best_params(row['best_params'])

        configs = {
            'model': row['model'],
            'scaler': row['scaler'],
            'sampler': row['balancing_name'],
            'params': params
        }
        all_configs.append(configs)
    return all_configs


def evaluate_valid(y_pred, y_proba, config, training_duration):
    return save_params_model_with_evaluate_valid_data(
        model=config["model"],
        scaler=config["scaler"],
        balancing_name=config["sampler"],
        training_time=training_duration,
        predictions=y_pred.tolist(),
        y_proba=y_proba.tolist()

    )


def evaluate_test(y_test, y_pred, y_proba, config, training_duration):
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


class ModelEvaluator:


    # def __train_and_predict(self, config, X_train, y_train, X_eval, preprocessing_file):
    #     scaler = self.__get_transformer_from_name(config["scaler"], "scaler")
    #     sampler = self.__get_transformer_from_name(config["sampler"], "sampler")
    #     pipe = self.create_pipeline()
    #     pipe.set_params(clf=MODELS[model_name], scaler=scaler, sampler=sampler, **config["params"])
    #
    #     start_time = time.time()
    #     pipe.fit(X_train, y_train)
    #     training_duration = time.time() - start_time
    #
    #     y_pred = pipe.predict(X_eval)
    #     y_proba = pipe.predict_proba(X_eval)[:, 1]
    #
    #     return y_pred, y_proba, training_duration

    def evaluate_to_valid_data(self, train_data, valid_data, results_df, preprocessing_file):
        configs = get_configs(results_df)
        X_train, y_train = prepare_data(train_data)
        X_valid, _ = prepare_data(valid_data)
        results = []
        for config in configs:
            y_pred, y_proba, duration = self.__train_and_predict(config, X_train, y_train, X_valid, preprocessing_file)
            result = evaluate_valid(y_pred, y_proba, config, duration)
            results.append(result)
        return results


    def evaluate_to_test_data(self, train_data, test_data, results_df, preprocessing_file):
        configs = get_configs(results_df)
        X_train, y_train = prepare_data(train_data)
        X_test, y_test = prepare_data(test_data)
        results = []
        for config in configs:
            y_pred, y_proba, duration = self.__train_and_predict(config, X_train, y_train, X_test, preprocessing_file)
            result = evaluate_test(y_test, y_pred, y_proba, config, duration)
            results.append(result)

        return results
    # TODO dodać printa
