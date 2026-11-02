import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .evaluation import evaluate_regression

BASELINES = {
    "Trung vị toàn bộ": (),
    "Trung vị theo hãng + dòng xe": (
        ("carbrand_name", "carmodel_name"),
        ("carbrand_name",),
    ),
    "Trung vị theo hãng + dòng xe + tuổi xe": (
        ("carbrand_name", "carmodel_name", "car_age"),
        ("carbrand_name", "carmodel_name"),
        ("carbrand_name",),
    ),
}


class GroupMedianRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, group_levels=()):
        self.group_levels = group_levels

    def fit(self, X, y):
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.overall_median_ = float(target.median())
        self.group_medians_ = []
        for columns in self.group_levels:
            medians = target.groupby([X[column] for column in columns]).median()
            if medians.index.nlevels == 1:
                medians.index = pd.MultiIndex.from_arrays([medians.index])
            self.group_medians_.append(medians)
        return self

    def predict(self, X):
        check_is_fitted(self, "overall_median_")
        predictions = pd.Series(np.nan, index=X.index)
        for columns, medians in zip(self.group_levels, self.group_medians_):
            keys = pd.MultiIndex.from_frame(X.loc[:, list(columns)])
            predictions = predictions.fillna(pd.Series(medians.reindex(keys).to_numpy(), index=X.index))
        return predictions.fillna(self.overall_median_).to_numpy()


def baseline_comparison(X_train, y_train, X_test, y_test, fitted_models):
    candidates = [
        (name, "Baseline không dùng ML", GroupMedianRegressor(levels).fit(X_train, y_train))
        for name, levels in BASELINES.items()
    ]
    candidates += [(name, "Mô hình ML", model) for name, model in fitted_models.items()]
    rows = []
    for name, kind, model in candidates:
        metrics = evaluate_regression(y_test, model.predict(X_test))
        rows.append({
            "Cách dự đoán": name,
            "Loại": kind,
            **{metric: metrics[metric] for metric in ("R2", "MAE", "RMSE")},
        })
    return pd.DataFrame(rows)
