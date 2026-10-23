"""Checks for saved predictions, input handling and fitted preprocessing."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from src.data import ROOT, split_data, load_data
from src.preprocessing import make_pipeline


def test_saved_model_roundtrip_and_raw_input(tmp_path):
    model = joblib.load(ROOT / 'models/amesvalue_pipeline.joblib')
    house = pd.read_csv(ROOT / 'data/example_house.csv', keep_default_na=False, na_values=[''])
    expected = model.predict(house)
    path = tmp_path / 'roundtrip.joblib'
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(house), expected)
    assert expected.shape == (1,)
    assert np.isfinite(expected).all()


def test_unknown_category_missing_values_and_schema():
    model = joblib.load(ROOT / 'models/amesvalue_pipeline.joblib')
    house = pd.read_csv(ROOT / 'data/example_house.csv', keep_default_na=False, na_values=[''])
    house['Neighborhood'] = 'Previously unseen neighborhood'
    house['LotFrontage'] = np.nan
    house['GarageType'] = np.nan
    assert np.isfinite(model.predict(house)).all()
    with pytest.raises(ValueError, match='Missing required columns'):
        model.predict(house.drop(columns='GrLivArea'))


def test_imputer_uses_only_fit_rows():
    X, _, y, _ = split_data(load_data())
    fit_X, fit_y = X.iloc[:80].copy(), y.iloc[:80]
    model = clone(make_pipeline(Ridge(solver='lsqr'), scale=True))
    model.fit(fit_X, fit_y)
    pre = model.named_steps['preprocess']
    num_names = pre.transformers_[0][2]
    imputer = pre.named_transformers_['numeric'].named_steps['impute']
    index = list(num_names).index('LotFrontage')
    assert imputer.statistics_[index] == pytest.approx(fit_X.LotFrontage.median())
    before = imputer.statistics_.copy()
    raw = X.iloc[80:81].copy()
    raw['LotFrontage'] = 1e9
    model.predict(raw)
    np.testing.assert_array_equal(imputer.statistics_, before)
