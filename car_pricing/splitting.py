from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLUMNS, TARGET_COLUMN, checked_feature_columns, prepare_model_features
from .validation import normalize_listing_ids, numeric_values, validate_columns


@dataclass(frozen=True)
class PricingSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_listing_ids: pd.Series
    test_listing_ids: pd.Series

    def summary(self):
        total = len(self.X_train) + len(self.X_test)
        return pd.DataFrame([
            {
                "Tập": name,
                "Số tin": len(features),
                "Tỷ lệ (%)": len(features) / total * 100,
                "Số đặc trưng": features.shape[1],
                "Giá trung vị (triệu VNĐ)": target.median(),
            }
            for name, features, target in (
                ("Train", self.X_train, self.y_train),
                ("Test", self.X_test, self.y_test),
            )
        ])


def split_pricing_data(dataset, test_size=0.2, random_state=42, columns=FEATURE_COLUMNS):
    columns = checked_feature_columns(columns)
    validate_columns(dataset, ("list_id", TARGET_COLUMN, *columns), "Dữ liệu mô hình")
    if not dataset.index.is_unique:
        raise ValueError("Index dữ liệu mô hình phải duy nhất.")
    features = prepare_model_features(dataset, columns)
    target = numeric_values(dataset[TARGET_COLUMN], TARGET_COLUMN)
    if target.le(0).any():
        raise ValueError("Giá phải lớn hơn 0.")
    identifiers = normalize_listing_ids(dataset["list_id"], "Dữ liệu mô hình")
    split = PricingSplit(*train_test_split(
        features, target, identifiers,
        test_size=test_size, random_state=random_state, shuffle=True,
    ))
    for X_part, y_part, ids in (
        (split.X_train, split.y_train, split.train_listing_ids),
        (split.X_test, split.y_test, split.test_listing_ids),
    ):
        if len(y_part) < 2 or y_part.nunique() < 2:
            raise ValueError("Mỗi tập cần ít nhất hai tin với giá khác nhau để đánh giá R².")
        if not X_part.index.equals(y_part.index) or not X_part.index.equals(ids.index):
            raise ValueError("Đặc trưng, giá và mã tin không khớp dòng sau khi chia.")
    if split.train_listing_ids.isin(split.test_listing_ids).any():
        raise ValueError("Có tin xuất hiện trong cả train và test.")
    return split
