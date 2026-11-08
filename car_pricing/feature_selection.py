import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score

from .features import CANDIDATE_CATEGORICAL_FEATURES, CANDIDATE_FEATURES, MISSING_LABEL, NUMERIC_FEATURES


def cv_r2_scores(pipeline, X, y, folds):
    scores = []
    for fit_rows, score_rows in folds:
        fitted = clone(pipeline).fit(X.iloc[fit_rows], y.iloc[fit_rows])
        scores.append(r2_score(y.iloc[score_rows], fitted.predict(X.iloc[score_rows])))
    return np.array(scores)


def mutual_information_report(X_train, y_train, random_state=42, columns=CANDIDATE_FEATURES):
    columns = list(columns)
    encoded = X_train.loc[:, columns].copy()
    for column in (c for c in columns if c in NUMERIC_FEATURES):
        encoded[column] = encoded[column].fillna(encoded[column].median())
    for column in (c for c in columns if c in CANDIDATE_CATEGORICAL_FEATURES):
        encoded[column] = encoded[column].fillna(MISSING_LABEL).astype("category").cat.codes
    scores = mutual_info_regression(
        encoded, np.log(y_train),
        discrete_features=[column not in NUMERIC_FEATURES for column in columns],
        random_state=random_state,
    )
    return pd.DataFrame({
        "Đặc trưng": columns,
        "Mutual information": scores,
        "Số giá trị": [X_train[column].nunique() for column in columns],
    }).sort_values("Mutual information", ascending=False, ignore_index=True)


def neutralize(features, columns):
    neutral = features.copy()
    for column in columns:
        if column in NUMERIC_FEATURES:
            neutral[column] = np.nan
        elif column in CANDIDATE_CATEGORICAL_FEATURES:
            neutral[column] = MISSING_LABEL
        else:
            neutral[column] = np.int8(0)
    return neutral


def drop_column_report(pipeline, X_train, y_train, candidates, cv):
    folds = list(cv.split(X_train))
    baseline = cv_r2_scores(pipeline, X_train, y_train, folds)
    rows = []
    for name, columns in candidates.items():
        loss = baseline - cv_r2_scores(pipeline, neutralize(X_train, columns), y_train, folds)
        rows.append({
            "Bỏ": name,
            "Số cột": len(columns),
            "R² CV khi bỏ": baseline.mean() - loss.mean(),
            "Mức giảm R²": loss.mean(),
            "Độ lệch chuẩn mức giảm": loss.std(ddof=1),
            "Số fold R² giảm": int((loss > 0).sum()),
        })
    report = pd.DataFrame(rows).sort_values("Mức giảm R²", ascending=False, ignore_index=True)
    report.attrs["baseline_r2"] = baseline.mean()
    return report


def selection_decision(report, n_folds):
    helps = report["Mức giảm R²"].gt(0) & report["Số fold R² giảm"].gt(n_folds / 2)
    return report.assign(**{"Gợi ý theo quy tắc": np.where(helps, "Giữ", "Loại")})
