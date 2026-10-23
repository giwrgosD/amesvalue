"""Predict sale prices from a CSV of house characteristics."""
import argparse
import joblib
import pandas as pd
from src.data import ROOT


def predict(csv_path):
    # Load only trusted joblib files: deserialization can execute Python code.
    model = joblib.load(ROOT / 'models/amesvalue_pipeline.joblib')
    houses = pd.read_csv(csv_path, keep_default_na=False, na_values=[''])
    return model.predict(houses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=str, default=str(ROOT / 'data/example_house.csv'))
    args = parser.parse_args()
    for value in predict(args.input):
        print(f'Predicted sale price: ${value:,.0f}')
