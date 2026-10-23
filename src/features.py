"""Check house inputs and optionally add bathroom and porch totals."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class HouseFeatures(TransformerMixin, BaseEstimator):
    """Preserve the input schema and optionally add two row-wise sums.

    No target, global averages, or other houses are used in transform().
    fit() only records the required column names, not preprocessing statistics.
    """
    def __init__(self, engineer=False):
        self.engineer = engineer

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, 'feature_names_in_')
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f'Missing required columns: {sorted(missing)}')
        result = X.loc[:, self.feature_names_in_].copy()
        # MSSubClass is a dwelling-type code, not a continuous measurement.
        result['MSSubClass'] = result['MSSubClass'].map(
            lambda value: str(int(value)) if not np.isnan(value) else np.nan)
        if self.engineer:
            # Half-baths count as 0.5. This is a count proxy, not floor area.
            result['TotalBathrooms'] = (result['FullBath'] + result['BsmtFullBath']
                + 0.5 * (result['HalfBath'] + result['BsmtHalfBath']))
            # Non-overlapping porch areas; wood decks are intentionally excluded.
            result['TotalPorchSF'] = (result['OpenPorchSF'] + result['EnclosedPorch']
                + result['3SsnPorch'] + result['ScreenPorch'])
            # Missing source values propagate; fold-local imputation handles them.
        return result
