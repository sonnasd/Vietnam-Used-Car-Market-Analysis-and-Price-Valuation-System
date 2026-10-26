from numbers import Integral

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from car_pricing.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURES


def build_preprocessor(min_model_frequency=10, dense_output=False):
    if (
        isinstance(min_model_frequency, bool)
        or not isinstance(min_model_frequency, Integral)
        or min_model_frequency < 1
    ):
        raise ValueError("Ngưỡng gom dòng xe hiếm phải là số nguyên dương.")

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Không rõ", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
        ]
    )
    model_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Không rõ", keep_empty_features=True)),
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
    other_categories = [column for column in CATEGORICAL_FEATURES if column != "carmodel_name"]
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, other_categories),
            ("car_model", model_pipeline, ["carmodel_name"]),
            ("text", "passthrough", list(TEXT_FEATURES)),
        ],
        sparse_threshold=0.0 if dense_output else 1.0,
        remainder="drop",
    )


def build_model_pipeline(model, min_model_frequency=10, dense_output=False):
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    min_model_frequency=min_model_frequency,
                    dense_output=dense_output,
                ),
            ),
            ("model", clone(model)),
        ]
    )
