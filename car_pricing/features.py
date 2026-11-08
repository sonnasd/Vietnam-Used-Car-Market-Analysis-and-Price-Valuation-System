from numbers import Integral

import numpy as np
import pandas as pd

from .validation import normalize_listing_ids, numeric_values, validate_columns

NUMERIC_FEATURES = ("car_age", "mileage_v2")

# Đặc trưng ứng viên: toàn bộ cột được phân tích ở mục 2.3 của notebook 05.
CANDIDATE_CATEGORICAL_FEATURES = (
    "carbrand_name",
    "carmodel_name",
    "gearbox",
    "fuel",
    "cartype",
    "carcolor",
    "carorigin",
    "carseats",
    "region_name",
)
CANDIDATE_TEXT_FEATURES = (
    "co_mo_ta",
    "vb_chinh_chu",
    "vb_bao_duong_hang",
    "vb_khong_dam_dung",
    "vb_khong_ngap_nuoc",
    "vb_zin",
    "vb_khong_tua_odo",
    "vb_bao_test",
    "vb_tra_gop",
    "vb_bao_hanh",
)
CANDIDATE_FEATURES = (*NUMERIC_FEATURES, *CANDIDATE_CATEGORICAL_FEATURES, *CANDIDATE_TEXT_FEATURES)

# Loại theo drop-column CV ở mục 2.3 của notebook 05: bỏ riêng từng đặc trưng này,
# R² CV không giảm ở quá nửa số fold. Notebook kiểm tra lại danh sách mỗi lần chạy.
DROPPED_FEATURES = ("carcolor", "co_mo_ta", "vb_bao_test", "vb_tra_gop", "vb_bao_hanh")

# Đặc trưng đưa vào mô hình (schema của Pipeline đã lưu).
CATEGORICAL_FEATURES = tuple(c for c in CANDIDATE_CATEGORICAL_FEATURES if c not in DROPPED_FEATURES)
TEXT_FEATURES = tuple(c for c in CANDIDATE_TEXT_FEATURES if c not in DROPPED_FEATURES)
FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *TEXT_FEATURES)
TARGET_COLUMN = "price_million"
MISSING_LABEL = "Không rõ"
FIRST_CAR_YEAR = 1886


def checked_feature_columns(columns):
    columns = tuple(columns)
    unknown = [column for column in columns if column not in CANDIDATE_FEATURES]
    if unknown or len(set(columns)) != len(columns):
        raise ValueError(f"Danh sách đặc trưng không hợp lệ: {unknown or 'có cột trùng'}.")
    return columns


def prepare_model_features(frame, columns=FEATURE_COLUMNS):
    columns = checked_feature_columns(columns)
    validate_columns(frame, columns, "Đặc trưng đầu vào")
    features = frame.loc[:, list(columns)].copy()
    for column in columns:
        if column in NUMERIC_FEATURES:
            features[column] = numeric_values(
                features[column], column, allow_missing=True, minimum=0,
                integer=column == "car_age",
            )
        elif column in CANDIDATE_CATEGORICAL_FEATURES:
            labels = features[column].astype("string").str.normalize("NFC").str.strip()
            labels = labels.mask(labels.eq(""))
            features[column] = labels.astype(object).where(labels.notna(), np.nan)
        else:
            flags = numeric_values(features[column], column)
            if not flags.isin((0, 1)).all():
                raise ValueError(f"Cờ {column} chỉ nhận 0 hoặc 1.")
            features[column] = flags.astype(np.int8)
    return features


def derive_car_age(cars, reference_year):
    if isinstance(reference_year, bool) or not isinstance(reference_year, Integral):
        raise ValueError("Năm tham chiếu phải là số nguyên.")
    years = numeric_values(cars["mfdate"], "mfdate", allow_missing=True, integer=True)
    if not years.dropna().between(FIRST_CAR_YEAR, reference_year).all():
        raise ValueError(f"Năm sản xuất nằm ngoài khoảng từ {FIRST_CAR_YEAR} đến năm tham chiếu.")
    ages = reference_year - years
    if "car_age" in cars:
        saved = numeric_values(cars["car_age"], "car_age", allow_missing=True, minimum=0)
        comparable = saved.notna() & ages.notna()
        if saved[comparable].ne(ages[comparable]).any():
            raise ValueError("car_age không khớp mfdate và năm tham chiếu.")
    return ages


def build_feature_dataset(cars, text_flags, reference_year=None):
    age_source = "car_age" if reference_year is None else "mfdate"
    validate_columns(
        cars, ("list_id", TARGET_COLUMN, age_source, "mileage_v2", *CANDIDATE_CATEGORICAL_FEATURES), "Dữ liệu xe",
    )
    validate_columns(text_flags, ("list_id", *CANDIDATE_TEXT_FEATURES), "Text flags")

    listing_ids = normalize_listing_ids(cars["list_id"], "Dữ liệu xe")
    features = cars.loc[:, ["mileage_v2", *CANDIDATE_CATEGORICAL_FEATURES]].copy()
    features["car_age"] = cars["car_age"] if reference_year is None else derive_car_age(cars, reference_year)
    features = features.join(_align_text_flags(text_flags, listing_ids))

    dataset = pd.DataFrame({"list_id": listing_ids, TARGET_COLUMN: _validated_target(cars)})
    return dataset.join(prepare_model_features(features, CANDIDATE_FEATURES), validate="one_to_one")


def _align_text_flags(text_flags, listing_ids):
    flag_ids = normalize_listing_ids(text_flags["list_id"], "Text flags")
    missing = (~listing_ids.isin(flag_ids)).sum()
    extra = (~flag_ids.isin(listing_ids)).sum()
    if missing or extra:
        raise ValueError(f"list_id giữa hai file không khớp: {missing} tin thiếu cờ, {extra} tin dư cờ.")
    flags = text_flags.loc[:, list(CANDIDATE_TEXT_FEATURES)].set_axis(flag_ids.to_numpy())
    return flags.reindex(listing_ids.to_numpy()).set_axis(listing_ids.index)


def _validated_target(cars):
    target = numeric_values(cars[TARGET_COLUMN], TARGET_COLUMN)
    if target.le(0).any():
        raise ValueError("Giá phải lớn hơn 0.")
    if "price" in cars:
        price = numeric_values(cars["price"], "price") / 1_000_000
        if not np.isclose(target, price, rtol=1e-9, atol=1e-6).all():
            raise ValueError("price_million không khớp price / 1_000_000.")
    return target
