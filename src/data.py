"""Load the Ames dataset, inspect it and create the train/test split."""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data/raw/ames_housing.csv'
SEED = 42
TARGET = 'SalePrice'
# Predict before the transaction: these describe its eventual outcome/date.
DROP_COLUMNS = ['Id', 'SaleType', 'SaleCondition', 'MoSold', 'YrSold']


def load_data(download=False):
    """Use the bundled CSV; explicitly download again with --download."""
    if download or not DATA_PATH.exists():
        housing = fetch_openml(data_id=42165, as_frame=True)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        housing.frame.to_csv(DATA_PATH, index=False)
    # Preserve the real category "None" (e.g. no masonry); only blanks are missing.
    return pd.read_csv(DATA_PATH, keep_default_na=False, na_values=[''])


def split_data(df):
    X = df.drop(columns=[TARGET] + DROP_COLUMNS)
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.20, random_state=SEED)


def inspect_data(df):
    """Basic inventory only. Full EDA is performed after splitting."""
    lines = [f'Shape: {df.shape}', f'Exact duplicate rows: {df.duplicated().sum()}',
             f'Duplicates excluding Id: {df.drop(columns="Id").duplicated().sum()}',
             '\nColumns and types:\n' + df.dtypes.to_string(),
             '\nRaw numeric/categorical counts (includes Id/target):\n' +
             str((len(df.select_dtypes('number').columns), len(df.select_dtypes('object').columns))),
             '\nMissing values:\n' + df.isna().sum().loc[lambda s: s > 0].to_string(),
             '\nTarget summary (USD):\n' + df[TARGET].describe().to_string(),
             '\nBasic target distribution: counts in five equal-width bins:\n' +
             df[TARGET].value_counts(bins=5, sort=False).to_string(),
             '\nRemoved: ' + ', '.join(DROP_COLUMNS)]
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    report = inspect_data(load_data(args.download))
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/data_overview.txt').write_text(report)
    print(report)
