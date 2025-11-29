import os
from datetime import datetime

import pandas as pd
from pathlib import Path
import yaml


def load_data(path):
    with open(path, "r") as f:
        file = pd.read_csv(f)
    return file


def load_config(path):
    with open(path, "r") as f:
        file = yaml.safe_load(f)
    return file


def save_params_model(model, scaler, training_time, accuracy_score_val, precision_score_val, recall_score_val,
                      f1_score_val, best_params=None):
    result = {
        "model": model,
        "scaler": scaler,
        "time_trening": training_time,  # Zmieniono nazwę klucza na ang/pol
        "accuracy_score": accuracy_score_val,
        "precision_score": precision_score_val,
        "recall_score": recall_score_val,
        "f1_score": f1_score_val,
        "best_params": best_params
    }
    return result


def to_dataframe(results_list):
    df = pd.DataFrame(results_list)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    file_path = os.path.join("results", f"results_{timestamp}.csv")

    df.to_csv(file_path, index=False)
    print(f"Wyniki zapisane do: {file_path}")
    return df


if __name__ == "__main__":
    root = Path.cwd().parent
    # d = load_data(f"{root}/data/test.csv")
    # print(d)
    c = load_config(f"config/training.yaml")
    print(c)
