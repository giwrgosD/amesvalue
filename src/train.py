"""Compare regression models with cross-validation and save the selected pipeline.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from src.data import ROOT, SEED, load_data, split_data
from src.preprocessing import make_pipeline

SCORING = {'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error', 'R2': 'r2'}


def summarize(name, scores, phase):
    row = {'Model': name, 'Phase': phase}
    for metric in SCORING:
        values = scores['test_' + metric] * (-1 if metric != 'R2' else 1)
        row[metric] = float(values.mean())
        row[metric + '_std'] = float(values.std(ddof=1))
    return row


def main():
    X_train, _, y_train, _ = split_data(load_data())
    # Exactly the same validation partitions for every model and experiment.
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train))
    reports = ROOT / 'reports'
    (reports / 'figures').mkdir(parents=True, exist_ok=True)
    (ROOT / 'models').mkdir(exist_ok=True)
    rows, candidates = [], {}

    def compare(name, estimator, phase):
        print(f'Cross-validation: {name}', flush=True)
        scores = cross_validate(estimator, X_train, y_train, cv=folds,
                                scoring=SCORING, n_jobs=1, error_score='raise')
        row = summarize(name, scores, phase)
        rows.append(row)
        candidates[name] = estimator
        print(f"  MAE={row['MAE']:.0f} RMSE={row['RMSE']:.0f} R2={row['R2']:.3f}", flush=True)
        return row

    # The median baseline ignores the house features.
    compare('Dummy median', DummyRegressor(strategy='median'), 4)
    compare('Linear Regression', make_pipeline(LinearRegression(), scale=True), 6)
    compare('Ridge', make_pipeline(Ridge(alpha=10, solver='lsqr'), scale=True), 7)
    compare('Decision Tree', make_pipeline(DecisionTreeRegressor(
        max_depth=8, min_samples_leaf=3, random_state=SEED)), 8)
    compare('Random Forest', make_pipeline(RandomForestRegressor(
        n_estimators=100, min_samples_leaf=2, random_state=SEED, n_jobs=2)), 8)
    pd.DataFrame(rows).to_csv(reports / 'results.csv', index=False)

    # Compare log targets for the two best initial models by CV RMSE.
    # TTR reverses the transform before the dollar-based scorers run.
    top_two = sorted(rows[1:], key=lambda row: row['RMSE'])[:2]
    variants = []
    for row in top_two:
        name = row['Model']
        variants.append(row)
        variants.append(compare(name + ' + log target', TransformedTargetRegressor(
            regressor=clone(candidates[name]), func=np.log1p, inverse_func=np.expm1), 10))

    # Try the bathroom and porch totals on the best variant.
    best = min(variants, key=lambda row: row['RMSE'])
    engineered = clone(candidates[best['Model']])
    prefix = 'regressor__' if isinstance(engineered, TransformedTargetRegressor) else ''
    engineered.set_params(**{prefix + 'features__engineer': True})
    compare(best['Model'] + ' + two features', engineered, 11)

    # Tune the best candidate; include its existing settings in the grid.
    best = min(rows[1:], key=lambda row: row['RMSE'])
    estimator = clone(candidates[best['Model']])
    wrapped = isinstance(estimator, TransformedTargetRegressor)
    pipeline = estimator.regressor if wrapped else estimator
    model = pipeline.named_steps['model']
    prefix = ('regressor__' if wrapped else '') + 'model__'
    if isinstance(model, Ridge):
        grid = {prefix + 'alpha': [1, 10, 100, 1000]}
    elif isinstance(model, RandomForestRegressor):
        grid = {prefix + 'max_features': [0.7, 1.0], prefix + 'min_samples_leaf': [1, 2, 4]}
    elif isinstance(model, DecisionTreeRegressor):
        grid = {prefix + 'max_depth': [5, 8, None], prefix + 'min_samples_leaf': [1, 3, 5]}
    else:
        # If Linear Regression wins, compare fits with and without an intercept.
        grid = {prefix + 'fit_intercept': [True, False]}
    print('Modest grid search:', grid, flush=True)
    search = GridSearchCV(estimator, grid, scoring=SCORING, refit=False,
                          cv=folds, n_jobs=1, error_score='raise')
    search.fit(X_train, y_train)
    cv = pd.DataFrame(search.cv_results_)
    cv.to_csv(reports / 'tuning_results.csv', index=False)
    index = int(cv['mean_test_RMSE'].idxmax())
    tuned = clone(estimator).set_params(**search.cv_results_['params'][index])
    tuned_row = {'Model': best['Model'] + ' (tuned)', 'Phase': 12}
    for metric in SCORING:
        # Calculate sample SD consistently with the earlier CV results.
        values = np.array([cv.loc[index, f'split{i}_test_{metric}'] for i in range(5)])
        values *= -1 if metric != 'R2' else 1
        tuned_row[metric] = float(values.mean())
        tuned_row[metric + '_std'] = float(values.std(ddof=1))
    rows.append(tuned_row)
    candidates[tuned_row['Model']] = tuned
    results = pd.DataFrame(rows)
    results.to_csv(reports / 'experiments.csv', index=False)

    # For candidates within 1% of the lowest CV RMSE,
    # prefer a linear family, no extra features, and no target transform.
    eligible = results[results.Model != 'Dummy median'].copy()
    eligible = eligible[eligible.RMSE <= eligible.RMSE.min() * 1.01]
    def simplicity(name):
        return (0 if name.startswith(('Ridge', 'Linear')) else
                1 if name.startswith('Decision Tree') else 2,
                'two features' in name, 'log target' in name)
    selected = min(eligible.to_dict('records'), key=lambda r: (simplicity(r['Model']), r['RMSE']))
    final_model = clone(candidates[selected['Model']])
    selection = {'selected_model': selected['Model'], 'cv_metrics': selected,
                 'tuning_best_params': search.cv_results_['params'][index],
                 'rule': 'Minimum CV RMSE; within 1%, prefer linear family, then no engineered features, then raw target.',
                 'reason': 'Selection uses training CV only. MAE, R2 and fold SD are saved for context; SD is not a confidence interval.',
                 'final_parameters': final_model.get_params()['regressor'].get_params()['model'].get_params()
                    if isinstance(final_model, TransformedTargetRegressor)
                    else final_model.named_steps['model'].get_params()}
    (reports / 'selection.json').write_text(json.dumps(selection, indent=2))
    # Fit the selected model on all training rows.
    final_model.fit(X_train, y_train)
    # Save preprocessing and the fitted model together.
    joblib.dump(final_model, ROOT / 'models/amesvalue_pipeline.joblib', compress=3)
    X_train.head(1).to_csv(ROOT / 'data/example_house.csv', index=False)
    base = results[results.Phase <= 8].sort_values('RMSE')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(base.Model, base.RMSE, xerr=base.RMSE_std, color='#287c8e')
    ax.set(xlabel='Mean validation RMSE (USD); bars show fold SD', title='Five-fold training CV')
    fig.tight_layout(); fig.savefig(reports / 'figures/model_comparison.png', dpi=150); plt.close(fig)
    print('Selected:', selected['Model'], flush=True)


if __name__ == '__main__':
    main()
