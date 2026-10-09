import numpy as np
import pandas as pd

CATEGORICAL_SEGMENTS = (
    "carbrand_name",
    "carmodel_name",
    "cartype",
    "gearbox",
    "fuel",
    "region_name",
)
BIN_SEGMENTS = {
    "car_age_group": (
        "car_age",
        (-np.inf, 5, 10, 15, np.inf),
        ("<5 năm", "5–<10 năm", "10–<15 năm", ">=15 năm"),
    ),
    "price_band": (
        "y_true",
        (-np.inf, 300, 600, 900, np.inf),
        ("<300 triệu", "300–<600 triệu", "600–<900 triệu", ">=900 triệu"),
    ),
    "mileage_group": (
        "mileage_v2",
        (-np.inf, 50_000, 100_000, 200_000, np.inf),
        ("<50.000 km", "50.000–<100.000 km", "100.000–<200.000 km", ">=200.000 km"),
    ),
}
METRIC_COLUMNS = (
    "y_true",
    "y_pred",
    "residual",
    "absolute_error",
    "percentage_error",
)


def _category_labels(values):
    labels = values.astype("string").str.strip().replace("", pd.NA)
    return labels.fillna("Thiếu dữ liệu")


def _bin_labels(values, edges, labels):
    numeric = pd.to_numeric(values, errors="raise")
    return pd.cut(numeric, edges, labels=labels, right=False).astype("string").fillna(
        "Thiếu dữ liệu"
    )


def _description_labels(values):
    numeric = pd.to_numeric(values, errors="raise")
    if not numeric.dropna().isin((0, 1)).all():
        raise ValueError("co_mo_ta phải có giá trị 0, 1 hoặc thiếu.")
    return numeric.map({0: "Không có mô tả", 1: "Có mô tả"}).fillna("Thiếu dữ liệu")


def _validate_inputs(error_table, X_train, y_train, min_test_size):
    if not isinstance(min_test_size, (int, np.integer)) or min_test_size < 1:
        raise ValueError("Số tin test tối thiểu phải là số nguyên dương.")
    features = {*CATEGORICAL_SEGMENTS, "car_age", "mileage_v2"}
    for name, frame, required in (
        ("Sai số test", error_table, features.union(METRIC_COLUMNS)),
        ("Đặc trưng train", X_train, features),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{name} thiếu cột: {', '.join(sorted(missing))}.")
        if frame.columns.duplicated().any() or not frame.index.is_unique:
            raise ValueError(f"{name} có cột hoặc chỉ mục trùng nhau.")
    metrics = error_table.loc[:, list(METRIC_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(metrics).all() or error_table["y_true"].le(0).any():
        raise ValueError("Giá test phải dương và các giá trị sai số phải hữu hạn.")
    if isinstance(y_train, pd.Series) and not y_train.index.equals(X_train.index):
        raise ValueError("y_train phải có chỉ mục và thứ tự giống X_train.")
    target = np.asarray(y_train, dtype=float)
    if target.ndim != 1 or len(target) != len(X_train):
        raise ValueError("y_train phải là vector có độ dài bằng X_train.")
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Giá train phải hữu hạn và lớn hơn 0.")
    return pd.Series(target, index=X_train.index)


def _summarize_feature(feature, test_labels, train_labels, errors, min_test_size, labels=()):
    values = errors.loc[:, list(METRIC_COLUMNS)].copy()
    values["segment"] = test_labels.to_numpy()
    values["squared_error"] = values["residual"].pow(2)
    values["within_20pct"] = values["residual"].abs().le(values["y_true"] * 0.2)
    groups = values.groupby("segment", sort=False)
    metrics = groups.agg(
        n_test=("y_true", "size"),
        MAE=("absolute_error", "mean"),
        RMSE=("squared_error", "mean"),
        bias=("residual", "mean"),
        median_absolute_error=("absolute_error", "median"),
        MAPE=("percentage_error", "mean"),
        median_percentage_error=("percentage_error", "median"),
        median_actual_price=("y_true", "median"),
        within_20pct=("within_20pct", "mean"),
    )
    segments = pd.Index(
        list(dict.fromkeys([*labels, *test_labels.tolist(), *train_labels.tolist()])),
        name="segment",
    )
    metrics = metrics.reindex(segments)
    metrics["n_test"] = metrics["n_test"].fillna(0).astype(int)
    metrics["n_train"] = train_labels.value_counts().reindex(segments, fill_value=0)
    metrics["RMSE"] = np.sqrt(metrics["RMSE"])
    metrics["within_20pct"] *= 100
    metrics["eligible"] = metrics["n_test"].ge(min_test_size)
    metrics.insert(0, "feature", feature)
    return metrics.reset_index()


def build_segment_report(
    error_table, X_train, y_train, min_test_size=30, min_model_frequency=10
):
    if not isinstance(min_model_frequency, (int, np.integer)) or min_model_frequency < 1:
        raise ValueError("Tần suất dòng xe tối thiểu phải là số nguyên dương.")
    training_target = _validate_inputs(error_table, X_train, y_train, min_test_size)
    training = X_train.assign(y_true=training_target)
    reports = []
    for feature in CATEGORICAL_SEGMENTS:
        reports.append(
            _summarize_feature(
                feature,
                _category_labels(error_table[feature]),
                _category_labels(training[feature]),
                error_table,
                min_test_size,
            )
        )
    for feature, (column, edges, labels) in BIN_SEGMENTS.items():
        reports.append(
            _summarize_feature(
                feature,
                _bin_labels(error_table[column], edges, labels),
                _bin_labels(training[column], edges, labels),
                error_table,
                min_test_size,
                labels,
            )
        )
    if "co_mo_ta" in error_table and "co_mo_ta" in training:
        reports.append(
            _summarize_feature(
                "co_mo_ta",
                _description_labels(error_table["co_mo_ta"]),
                _description_labels(training["co_mo_ta"]),
                error_table,
                min_test_size,
                ("Không có mô tả", "Có mô tả"),
            )
        )

    model_labels = (
        "Chưa thấy trong train",
        f"Dòng xe hiếm (<{min_model_frequency} tin train)",
        f"Dòng xe từ {min_model_frequency} tin train",
    )
    train_models = _category_labels(training["carmodel_name"])
    train_counts = train_models.value_counts()
    frequency_groups = []
    for frame in (error_table, training):
        counts = _category_labels(frame["carmodel_name"]).map(train_counts).fillna(0)
        groups = np.select(
            (counts.eq(0), counts.lt(min_model_frequency)),
            model_labels[:2],
            default=model_labels[2],
        )
        frequency_groups.append(pd.Series(groups, index=frame.index))
    reports.append(
        _summarize_feature(
            "model_frequency_group",
            *frequency_groups,
            error_table,
            min_test_size,
            model_labels,
        )
    )
    columns = (
        "feature",
        "segment",
        "n_test",
        "n_train",
        "MAE",
        "RMSE",
        "bias",
        "median_absolute_error",
        "MAPE",
        "median_percentage_error",
        "median_actual_price",
        "within_20pct",
        "eligible",
    )
    return pd.concat(reports, ignore_index=True).loc[:, list(columns)]
