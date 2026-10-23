"""Build preprocessing and regression steps in one pipeline."""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features import HouseFeatures


def make_pipeline(model, scale=False, engineer=False):
    numeric_steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        numeric_steps.append(('scale', StandardScaler()))
    categorical = Pipeline([
        # Keep missing categorical values as a separate level.
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing_or_absent',
                                keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    preprocess = ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), make_column_selector(dtype_include='number')),
        ('categorical', categorical, make_column_selector(dtype_exclude='number'))],
        sparse_threshold=1.0)
    return Pipeline([('features', HouseFeatures(engineer=engineer)),
                     ('preprocess', preprocess), ('model', model)])
