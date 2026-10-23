# Workflow notes

## Loading and splitting

`src/data.py` loads the bundled CSV. The optional download uses OpenML dataset
42165. The loader preserves the literal category `None` and treats empty fields
as missing. `inspect_data()` produces a basic inventory of the full CSV, including
target summaries. Detailed exploration uses only the training subset.

`split_data()` removes the target, identifier and four transaction fields, then
creates an 80/20 random split with seed 42. The same CSV row order is needed to
reproduce the split. It contains 1,168 training and 292 test houses.

## Exploring the training data

The notebook looks at prices, numeric relationships, category distributions,
missing values and unusually large houses. Prices are right-skewed: the training
mean is about $181,442 and the median is $165,000. This led to comparing a log
transformation with the original target.

All rows are retained. An unusually large house is not necessarily a bad record.
Quality and living area have strong positive correlations with price, but these
relationships alone do not establish cause and effect.

## Preprocessing

`make_pipeline()` combines `HouseFeatures`, a `ColumnTransformer`, and a model.
Keeping them together means cross-validation fits preprocessing on each fold's
training rows and uses those fitted values on its validation rows.

- Numeric values: median imputation, plus standard scaling for linear models.
- Categorical values: a `Missing_or_absent` label, then one-hot encoding.
- Unseen categories: encoded as zeros in that feature's one-hot block.
- `MSSubClass`: converted to a string because it represents a dwelling category.

`HouseFeatures.fit()` records the input column names. `transform()` checks required
columns, restores their order, and optionally adds `TotalBathrooms` and
`TotalPorchSF`. These are row-level sums and do not use the target or other houses.
Missing components remain missing until imputation.

Total bathrooms counts full bathrooms plus half the number of half-bathrooms,
including basement bathrooms. Total porch area adds open, enclosed, three-season
and screen porch areas. The original component columns remain in the model.

## Comparing models

`src/train.py` uses identical shuffled five-fold partitions for all candidates.
The median baseline provides a reference. Linear Regression and Ridge test linear
relationships; Ridge adds regularization. The tree and forest allow nonlinear
relationships and interactions. Numeric scaling is unnecessary for trees.

MAE and RMSE are reported in dollars. RMSE is the selection metric because it
penalizes large errors more heavily. Scikit-learn returns negative loss scores;
the script changes their signs for reporting. Each reported CV metric is the
mean of fold scores, with a sample standard deviation across the five folds.
That standard deviation is not a confidence interval.

The script then compares log targets for the two best initial models.
`TransformedTargetRegressor` handles `log1p` during fitting and `expm1` during
prediction, so evaluation still uses dollars. This inverse transformation does
not include a correction for retransformation bias.

Next, the two engineered features are tested together on the best raw/log
variant. Only the strongest candidate is tuned. The supplied run searches six
forest combinations: `max_features` of 0.7 or 1.0 and `min_samples_leaf` of 1, 2 or 4.
There is no separate tuned forest experiment without the engineered features.

## Selecting and saving

Candidates within 1% of the minimum CV RMSE are eligible. The rule prefers linear
models, then a single tree, then a forest. Within that ordering it prefers no extra
features and no log target; remaining ties use RMSE.

The selected model is fitted on all training rows and saved as
`models/amesvalue_pipeline.joblib`. It contains preprocessing as well as the
regressor. The example CSV is the first training row.

The result files retain a `Phase` column from the experiment sequence:

| Phase | Experiment |
| --- | --- |
| 4 | Median baseline |
| 6 | Linear Regression |
| 7 | Ridge |
| 8 | Decision Tree and Random Forest |
| 10 | Log-target comparisons |
| 11 | Two engineered features |
| 12 | Hyperparameter tuning |

These identifiers are report labels, not additional processing steps.

## Test evaluation and interpretation

`src/evaluate.py` loads the fitted pipeline and predicts the test rows. The same
predictions feed all metrics and residual reports. Residuals are actual minus
predicted price, so positive residuals mean underprediction.

The price-band table groups houses by their actual test prices. It helps describe
errors after evaluation; those bands would not be available before a sale.
Feature importance comes from the fitted forest's impurity reductions. Correlated
inputs can share importance, and high-cardinality inputs can be favored.

Repeated model choices on the same CV folds introduce selection optimism. Training
EDA also informs those choices. The separate test provides a final assessment,
but rerunning it after changing the approach would not provide a fresh test.

## Prediction and tests

`src.predict` reads a raw CSV and passes it to the saved pipeline. All required
predictor columns must be present. It handles missing values and unseen categories,
but does not provide full input validation for malformed values or impossible
measurements.

The three existing tests check model serialization and raw input, missing values
and unseen categories, missing-column errors, and whether the imputer learns only
from fitted rows. They do not test every input type or reproduce the full training
experiment.
