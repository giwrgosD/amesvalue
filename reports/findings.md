# Results and error analysis

## Training experiments

| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Dummy median | $54,586 | $78,860 | -0.046 |
| Linear Regression | $19,892 | $40,668 | 0.688 |
| Ridge | $18,295 | $31,751 | 0.827 |
| Decision Tree | $25,377 | $39,380 | 0.739 |
| Random Forest | $18,302 | $30,788 | 0.838 |
| Random Forest + log target | $18,436 | $31,245 | 0.834 |
| Ridge + log target | $17,048 | $40,381 | 0.646 |
| Random Forest + two features | $18,220 | $30,706 | 0.839 |
| Random Forest + two features (tuned) | $17,720 | $29,833 | 0.848 |

MAE/RMSE are original-scale dollars even for log targets. All rows use the same
five training folds. Ridge improves on ordinary least squares, consistent with
regularization helping with correlated predictors. The forest improves RMSE over
both linear models and the individual tree, consistent with modeling nonlinear
relationships while averaging tree variance. These explanations are plausible
mechanisms, not experimentally isolated causes.

The raw forest's RMSE is $30,788. Log target worsens it to $31,245. Log Ridge achieves
lower typical absolute error (MAE $17,048), but RMSE $40,381 with fold SD about
$28,384 indicates very unstable large errors. I used RMSE for selection, so the lower MAE alone was not enough to choose log Ridge.

Adding the two sums barely improves the untuned forest ($30,788 to $30,706). Small
tuning then reaches $29,833 with max_features=0.7 and min_samples_leaf=2. Only this candidate was tuned. There is no separate tuned
comparison without the extra features, so this does not prove the engineered features caused the
full improvement. The 1% simplicity rule selects the tuned forest, whose RMSE is
more than 1% below simpler available candidates. MAE also improves to $17,720,
and its RMSE fold SD ($5,505) is lower than Ridge's ($8,087), although it remains
substantial. The forest sacrifices direct coefficient interpretation for stronger
CV RMSE; impurity importance supplies a limited descriptive explanation.

## Final test

MAE $17,197.05, RMSE $29,205.70, R² 0.8888.
The mean actual target is $178,839.81; median
$154,150.00. MAE is the average absolute miss, not an
interval into which every house's error falls. R² is not classification accuracy.
A higher test R² than mean CV R² can arise because the test sample and its price
variance differ; it is not evidence that testing improved the fitted model.

## Price-band residuals

Residual = actual − predicted. Positive means underprediction.

| price_band | count | mean_price | MAE | mean_residual |
| --- | --- | --- | --- | --- |
| Q1 low | 75 | $101,607 | $14,146 | $-12,012 |
| Q2 | 71 | $139,673 | $9,030 | $-691 |
| Q3 | 73 | $179,376 | $13,282 | $-1 |
| Q4 high | 73 | $295,746 | $32,190 | $19,476 |

Groups are defined descriptively by actual-price quartiles after test
prediction. Ties mean group counts need not be exactly equal. These groups are
not deployment-time filters and were not used to tune or engineer anything.
High-priced houses have much greater absolute error; low-priced houses are
also harder than the middle groups. The positive top-quartile residual supports
underprediction of expensive houses on average, not of every expensive house.

## Five large-error examples

`csv_row_index` is the zero-based original CSV row position, not a predictor.

| csv_row_index | Neighborhood | OverallQual | GrLivArea | actual | predicted | residual |
| --- | --- | --- | --- | --- | --- | --- |
| 691 | NoRidge | 10 | 4316 | $755,000 | $548,084 | $206,916 |
| 898 | NridgHt | 9 | 2364 | $611,657 | $436,965 | $174,692 |
| 1046 | StoneBr | 9 | 2868 | $556,581 | $413,681 | $142,900 |
| 218 | Crawfor | 7 | 1954 | $311,500 | $205,725 | $105,775 |
| 581 | NridgHt | 8 | 2042 | $253,293 | $357,419 | $-104,126 |

The first three rows are high-quality, expensive homes. The largest miss is a
4,316-square-foot, quality-10 home sold for $755,000. Scarce upper-tail examples
and forest averaging are plausible contributors to underprediction. Another
NridgHt home is overpredicted by about $104,126 despite quality 8: observed
features cannot explain every negotiated transaction. Unrecorded condition or
transaction context could matter, but the dataset does not verify a cause.
The older Crawfor home is also substantially underpredicted. These cases remain in the evaluation. They were not used for further tuning.

## Importance

The leading encoded features are OverallQual (about 42.6% of total impurity
importance), GrLivArea (12.9%), GarageCars (6.6%), and TotalBsmtSF (3.3%).
These shares are not percentages of price explained or causal contributions.
Correlated area measures divide credit, and continuous/high-cardinality features
can be favored. The top-15 plot shows exact transformed names from the fitted
ColumnTransformer. TotalBathrooms appears among these features; that alone does
not establish incremental value beyond the original bathroom counts.
