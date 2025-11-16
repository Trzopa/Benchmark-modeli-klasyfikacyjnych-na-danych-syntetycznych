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



if __name__ == "__main__":
    root = Path.cwd().parent
    # d = load_data(f"{root}/data/test.csv")
    # print(d)
    c = load_config(f"config/training.yaml")
    print(c)