"""Evaluate the saved model and report test errors and feature importance.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from src.data import ROOT, load_data, split_data


def main():
    _, X_test, _, y_test = split_data(load_data())
    model = joblib.load(ROOT / 'models/amesvalue_pipeline.joblib')
    predictions = model.predict(X_test)
    reports = ROOT / 'reports'
    figures = reports / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    metrics = {'MAE': float(mean_absolute_error(y_test, predictions)),
               'RMSE': float(root_mean_squared_error(y_test, predictions)),
               'R2': float(r2_score(y_test, predictions)),
               'test_target_mean': float(y_test.mean()),
               'test_target_median': float(y_test.median()), 'test_rows': len(y_test)}
    (reports / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    errors = X_test.copy()
    errors.insert(0, 'csv_row_index', X_test.index)
    errors['actual'] = y_test
    errors['predicted'] = predictions
    # Positive residual = underprediction; negative residual = overprediction.
    errors['residual'] = errors.actual - errors.predicted
    errors['absolute_error'] = errors.residual.abs()
    errors.to_csv(reports / 'test_predictions.csv', index=False)
    errors.nlargest(10, 'absolute_error').to_csv(reports / 'largest_errors.csv', index=False)
    # Group errors by actual price to compare cheaper and more expensive houses.
    errors['price_band'] = pd.qcut(errors.actual, 4, labels=['Q1 low', 'Q2', 'Q3', 'Q4 high'])
    bands = errors.groupby('price_band', observed=True).agg(
        count=('actual', 'size'), mean_price=('actual', 'mean'),
        MAE=('absolute_error', 'mean'), mean_residual=('residual', 'mean'))
    bands.to_csv(reports / 'error_by_price_band.csv')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    low, high = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    axes[0].scatter(y_test, predictions, alpha=0.55, s=18)
    axes[0].plot([low, high], [low, high], '--', color='black')
    axes[0].set(xlabel='Actual price (USD)', ylabel='Predicted price (USD)', title='Actual vs predicted')
    axes[1].scatter(predictions, errors.residual, alpha=0.55, s=18)
    axes[1].axhline(0, linestyle='--', color='black')
    axes[1].set(xlabel='Predicted price (USD)', ylabel='Actual − predicted (USD)', title='Residuals')
    axes[2].hist(errors.residual, bins=30, color='#287c8e')
    axes[2].axvline(0, linestyle='--', color='black')
    axes[2].set(xlabel='Residual (USD)', ylabel='Count', title='Residual distribution')
    # Compact tick labels keep the dollar axes readable.
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value / 1000:,.0f}k'))
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value / 1000:,.0f}k'))
    fig.tight_layout(); fig.savefig(figures / 'residual_analysis.png', dpi=160); plt.close(fig)

    # TTR's fitted inner pipeline is regressor_, not the unfitted regressor.
    pipeline = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    names = pipeline.named_steps['preprocess'].get_feature_names_out()
    fitted = pipeline.named_steps['model']
    if hasattr(fitted, 'feature_importances_'):
        values = fitted.feature_importances_
        label = 'Impurity importance (not causal)'
    else:
        values = np.ravel(fitted.coef_)
        label = 'Signed coefficient (model target scale)'
    importance = pd.DataFrame({'feature': names, 'value': values})
    importance['absolute_value'] = importance.value.abs()
    importance = importance.sort_values('absolute_value', ascending=False)
    importance.to_csv(reports / 'feature_importance.csv', index=False)
    top = importance.head(15).sort_values('absolute_value')
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature, top.value, color='#287c8e')
    ax.set(xlabel=label, title='Top 15 transformed features')
    fig.tight_layout(); fig.savefig(figures / 'feature_importance.png', dpi=150); plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print('\nErrors by actual-price quartile:\n', bands.to_string())


if __name__ == '__main__':
    main()
