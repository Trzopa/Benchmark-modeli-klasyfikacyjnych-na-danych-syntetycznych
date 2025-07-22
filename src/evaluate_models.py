import os
import time
import joblib
import pandas as pd
from sklearn import set_config
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

set_config(transform_output="pandas")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results')
models_dir = os.path.join(results_dir, 'artifacts')
metrics_dir = os.path.join(results_dir, 'metrics')
predictions_dir = os.path.join(results_dir, 'predictions')

os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

model_names = [
    'DecisionTreeClassifier',
    'KNeighborsClassifier',
    'LightGBM',
    'LogisticRegression',
    'RandomForestClassifier'
]
scalers = ['MinMaxScaler', 'NoScaling', 'StandardScaler']

path_test = os.path.join(data_dir, 'test.csv')
path_valid = os.path.join(data_dir, 'valid.csv')

test = pd.read_csv(path_test)
X_test = test.drop('target', axis=1)
y_test = test['target']
X_valid = pd.read_csv(path_valid)

train_metrics_path = os.path.join(results_dir, "metrics", "train_metrics.csv")
fit_times_df = pd.read_csv(train_metrics_path) if os.path.exists(train_metrics_path) else pd.DataFrame()

metrics = []

for model_name in model_names:
    for scaler_name in scalers:
        model_filename = f'{model_name}_{scaler_name}.pkl'
        model_path = os.path.join(models_dir, model_filename)

        if not os.path.exists(model_path):
            print(f"Brak modelu: {model_path}, pomijam.")
            continue

        model = joblib.load(model_path)

        if hasattr(model, "feature_names_in_"):
            X_test_eval = X_test[model.feature_names_in_]
            X_valid_eval = X_valid[model.feature_names_in_]
        else:
            X_test_eval = X_test
            X_valid_eval = X_valid

        start = time.time()
        y_pred = model.predict(X_test_eval)
        y_proba = model.predict_proba(X_test_eval)[:, 1] if hasattr(model, "predict_proba") else None
        score_time = round(time.time() - start, 4)

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Zapis metryk
        metrics.append({
            'model': model_name,
            'scaler': scaler_name,
            'accuracy': round(acc, 3),
            'roc_auc': round(roc, 3) if roc is not None else None,
            'precision': round(pre, 3),
            'recall': round(rec, 3),
            'f1': round(f1, 3),
            'score_time': score_time
        })

        valid_pred = model.predict(X_valid_eval)
        valid_proba = model.predict_proba(X_valid_eval)[:, 1] if hasattr(model, "predict_proba") else None

        valid_df = pd.DataFrame({
            'prediction': valid_pred,
            'probability': valid_proba
        })
        valid_df['probability'] = valid_df['probability'].round(3)
        valid_df.to_csv(
            os.path.join(predictions_dir, f'valid_predictions_{model_name}_{scaler_name}.csv'),
            index=False
        )

        test_df = pd.DataFrame({
            'true': y_test,
            'prediction': y_pred,
            'probability': y_proba
        })
        test_df['probability'] = test_df['probability'].round(3)
        test_df.to_csv(
            os.path.join(predictions_dir, f'test_predictions_{model_name}_{scaler_name}.csv'),
            index=False
        )

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(metrics_dir, 'test_metrics.csv'), index=False)

print("Evaluation completed.")
