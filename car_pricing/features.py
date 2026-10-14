import numpy as np
import pandas as pd

NUMERIC_FEATURES = ("car_age", "mileage_v2")
CATEGORICAL_FEATURES = (
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
TEXT_FEATURES = (
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
FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *TEXT_FEATURES)
TARGET_COLUMN = "price_million"


def _validate_columns(frame, columns, name):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise KeyError(f"{name} thiếu cột: {', '.join(sorted(missing))}.")
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} có tên cột trùng nhau.")


def _validate_listing_ids(frame, name):
    if frame["list_id"].isna().any():
        raise ValueError(f"{name} có list_id bị thiếu.")
    if frame["list_id"].duplicated().any():
        raise ValueError(f"{name} có list_id trùng nhau.")


def build_feature_dataset(cars, text_flags):
    source_columns = ("list_id", TARGET_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
    _validate_columns(cars, source_columns, "Dữ liệu xe")
    _validate_columns(text_flags, ("list_id", *TEXT_FEATURES), "Cờ văn bản")
    _validate_listing_ids(cars, "Dữ liệu xe")
    _validate_listing_ids(text_flags, "Cờ văn bản")

    missing_flags = ~cars["list_id"].isin(text_flags["list_id"])
    extra_flags = ~text_flags["list_id"].isin(cars["list_id"])
    if missing_flags.any() or extra_flags.any():
        raise ValueError(
            f"list_id giữa hai file không khớp: {missing_flags.sum()} tin thiếu cờ, "
            f"{extra_flags.sum()} tin dư cờ."
        )

    dataset = cars.loc[:, list(source_columns)].copy()
    target = pd.to_numeric(dataset[TARGET_COLUMN], errors="raise")
    if target.isna().any() or not np.isfinite(target).all() or target.le(0).any():
        raise ValueError("Giá phải hữu hạn, không thiếu và lớn hơn 0.")
    dataset[TARGET_COLUMN] = target

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(dataset[column], errors="raise")
        if not np.isfinite(values.dropna()).all():
            raise ValueError(f"Đặc trưng {column} có giá trị không hữu hạn.")
        dataset[column] = values

    for column in CATEGORICAL_FEATURES:
        values = dataset[column].astype("string").str.strip().replace("", pd.NA)
        dataset[column] = values.astype(object).where(values.notna(), np.nan)

    aligned_flags = text_flags.set_index("list_id").reindex(dataset["list_id"])
    for column in TEXT_FEATURES:
        values = pd.to_numeric(aligned_flags[column], errors="raise")
        if not values.isin((0, 1)).all():
            raise ValueError(f"Cờ {column} phải có giá trị 0 hoặc 1 và không được thiếu.")
        dataset[column] = values.to_numpy(dtype=np.int8)

    return dataset
