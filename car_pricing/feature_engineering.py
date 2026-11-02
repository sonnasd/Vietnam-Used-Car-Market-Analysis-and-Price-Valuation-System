import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_predict

from .error_segments import BIN_SEGMENTS
from .evaluation import evaluate_regression
from .feature_reports import FEATURE_GROUPS
from .feature_selection import cv_r2_scores
from .features import CATEGORICAL_FEATURES, MISSING_LABEL, NUMERIC_FEATURES
from .preprocessing import build_model_pipeline

IQR_FACTOR = 1.5


def km_per_year(features):
    return features["mileage_v2"] / features["car_age"].clip(lower=1)


CANDIDATE_FEATURES = {"km_moi_nam": km_per_year}


def missing_value_report(X_train):
    rows = []
    for group, columns in FEATURE_GROUPS.items():
        for column in columns:
            values = X_train[column]
            if column in NUMERIC_FEATURES:
                strategy, fill_value = "Trung vị của train", values.median()
            elif column in CATEGORICAL_FEATURES:
                strategy, fill_value = "Nhãn riêng cho giá trị thiếu", MISSING_LABEL
            else:
                strategy, fill_value = "Không cần: cờ luôn là 0/1", "-"
            rows.append({
                "Nhóm": group,
                "Đặc trưng": column,
                "Số thiếu": int(values.isna().sum()),
                "Tỷ lệ thiếu (%)": values.isna().mean() * 100,
                "Cách xử lý": strategy,
                "Giá trị điền": fill_value,
            })
    return pd.DataFrame(rows)


def categorical_encoding_report(X_train, min_model_frequency):
    rows = []
    for column in CATEGORICAL_FEATURES:
        counts = X_train[column].fillna(MISSING_LABEL).value_counts()
        rare = counts.lt(min_model_frequency)
        grouped = column == "carmodel_name"
        rows.append({
            "Đặc trưng": column,
            "Số nhãn trên train": len(counts),
            "Nhãn phổ biến nhất": counts.index[0],
            "Tỷ lệ nhãn phổ biến nhất (%)": counts.iloc[0] / counts.sum() * 100,
            f"Số nhãn dưới {min_model_frequency} tin": int(rare.sum()),
            "Cách mã hóa": (
                f"One-hot, gom nhãn dưới {min_model_frequency} tin thành một cột" if grouped else "One-hot"
            ),
        })
    return pd.DataFrame(rows)


def numeric_distribution_report(X_train, y_train):
    columns = {**{column: X_train[column] for column in NUMERIC_FEATURES}, y_train.name: y_train}
    return pd.DataFrame([
        {
            "Biến": name,
            "Nhỏ nhất": values.min(),
            "Trung vị": values.median(),
            "Lớn nhất": values.max(),
            "Độ lệch (skew)": values.skew(),
            "Biến đổi": "Giữ nguyên đơn vị triệu VNĐ" if name == y_train.name else "StandardScaler",
        }
        for name, values in columns.items()
    ])


def log_target_report(estimator, X_train, y_train, cv, **pipeline_options):
    log_estimator = TransformedTargetRegressor(regressor=clone(estimator), func=np.log, inverse_func=np.exp)
    _, edges, labels = BIN_SEGMENTS["price_band"]
    price_bands = pd.cut(y_train, edges, labels=labels, right=False)
    rows = []
    for name, model in (("Giá gốc", estimator), ("log(giá)", log_estimator)):
        pipeline = build_model_pipeline(model, **pipeline_options)
        predictions = pd.Series(cross_val_predict(pipeline, X_train, y_train, cv=cv), index=y_train.index)
        metrics = evaluate_regression(y_train, predictions)
        percentage_errors = (predictions - y_train).abs() / y_train * 100
        rows.append({
            "Biến mục tiêu": name,
            **{metric: metrics[metric] for metric in ("R2", "MAE", "RMSE")},
            "MAPE (%)": percentage_errors.mean(),
            **{
                f"MAPE {band} (%)": percentage_errors[price_bands == band].mean()
                for band in labels
            },
        })
    return pd.DataFrame(rows)


def new_feature_report(estimator, X_train, y_train, cv, candidates=CANDIDATE_FEATURES, **pipeline_options):
    folds = list(cv.split(X_train))
    baseline = cv_r2_scores(build_model_pipeline(estimator, **pipeline_options), X_train, y_train, folds)
    rows = []
    for name, make_feature in candidates.items():
        pipeline = build_model_pipeline(estimator, extra_numeric=(name,), **pipeline_options)
        X_new = X_train.assign(**{name: make_feature(X_train)})
        gain = cv_r2_scores(pipeline, X_new, y_train, folds) - baseline
        rows.append({
            "Đặc trưng mới": name,
            "R² CV gốc": baseline.mean(),
            "R² CV khi thêm": baseline.mean() + gain.mean(),
            "Mức tăng R²": gain.mean(),
            "Độ lệch chuẩn mức tăng": gain.std(ddof=1),
            "Số fold R² tăng": int((gain > 0).sum()),
        })
    return pd.DataFrame(rows)


def outlier_report(X_train, y_train):
    columns = {
        **{column: X_train[column] for column in NUMERIC_FEATURES},
        "km_moi_nam": km_per_year(X_train),
        y_train.name: y_train,
    }
    rows = []
    for name, values in columns.items():
        q1, q3 = values.quantile([0.25, 0.75])
        lower, upper = q1 - IQR_FACTOR * (q3 - q1), q3 + IQR_FACTOR * (q3 - q1)
        outside = values.lt(lower) | values.gt(upper)
        rows.append({
            "Biến": name,
            "Cận dưới IQR": lower,
            "Cận trên IQR": upper,
            "Số tin ngoài khoảng": int(outside.sum()),
            "Tỷ lệ (%)": outside.mean() * 100,
        })
    return pd.DataFrame(rows)


def implausible_listing_report(X_train):
    yearly = km_per_year(X_train)
    checks = {
        "ODO bằng 0 nhưng xe trên 2 năm tuổi": X_train["mileage_v2"].eq(0) & X_train["car_age"].gt(2),
        "Trên 100.000 km mỗi năm": yearly.gt(100_000),
    }
    return pd.DataFrame([
        {"Kiểm tra": name, "Số tin": int(mask.sum()), "Tỷ lệ (%)": mask.mean() * 100}
        for name, mask in checks.items()
    ])
