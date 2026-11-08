from numbers import Integral

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from car_pricing.features import (
    CANDIDATE_CATEGORICAL_FEATURES,
    CANDIDATE_TEXT_FEATURES,
    FEATURE_COLUMNS,
    MISSING_LABEL,
    NUMERIC_FEATURES,
    checked_feature_columns,
)


def build_preprocessor(min_model_frequency=10, dense_output=False, extra_numeric=(), features=FEATURE_COLUMNS):
    """``features`` mặc định là các đặc trưng đưa vào mô hình; mục 2.3 của notebook 05
    truyền toàn bộ đặc trưng ứng viên để phân tích trước khi chọn."""
    if (
        isinstance(min_model_frequency, bool)
        or not isinstance(min_model_frequency, Integral)
        or min_model_frequency < 1
    ):
        raise ValueError("Ngưỡng gom dòng xe hiếm phải là số nguyên dương.")
    features = checked_feature_columns(features)

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_LABEL, keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
        ]
    )
    model_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_LABEL, keep_empty_features=True)),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_model_frequency,
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )
    numeric = [column for column in features if column in NUMERIC_FEATURES]
    other_categories = [
        column for column in features
        if column in CANDIDATE_CATEGORICAL_FEATURES and column != "carmodel_name"
    ]
    car_model = [column for column in features if column == "carmodel_name"]
    text = [column for column in features if column in CANDIDATE_TEXT_FEATURES]
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, [*numeric, *extra_numeric]),
            ("categorical", categorical_pipeline, other_categories),
            ("car_model", model_pipeline, car_model),
            ("text", "passthrough", text),
        ],
        sparse_threshold=0.0 if dense_output else 1.0,
        remainder="drop",
    )


def build_model_pipeline(model, min_model_frequency=10, dense_output=False, extra_numeric=(), features=FEATURE_COLUMNS):
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    min_model_frequency=min_model_frequency,
                    dense_output=dense_output,
                    extra_numeric=extra_numeric,
                    features=features,
                ),
            ),
            ("model", clone(model)),
        ]
    )
