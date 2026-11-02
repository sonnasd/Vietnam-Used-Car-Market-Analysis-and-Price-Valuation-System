import numpy as np
import pandas as pd


def validate_columns(frame, columns, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} phải là DataFrame.")
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} có tên cột trùng nhau.")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise KeyError(f"{name} thiếu cột: {', '.join(sorted(missing))}.")
    if frame.empty:
        raise ValueError(f"{name} không được rỗng.")


def normalize_listing_ids(values, name):
    identifiers = values.astype("string").str.strip()
    if not identifiers.str.fullmatch(r"[0-9]+").fillna(False).all():
        raise ValueError(f"{name} có list_id thiếu hoặc không phải số nguyên dương.")
    try:
        identifiers = identifiers.astype("int64")
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"list_id trong {name} vượt giới hạn số nguyên 64 bit.") from exc
    if identifiers.le(0).any() or identifiers.duplicated().any():
        raise ValueError(f"{name} có list_id không dương hoặc trùng nhau.")
    return identifiers


def numeric_values(values, name, allow_missing=False, minimum=None, integer=False):
    try:
        result = pd.to_numeric(values, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} chứa giá trị không phải số.") from exc
    present = result.dropna()
    if not np.isfinite(present).all() or (not allow_missing and result.isna().any()):
        raise ValueError(f"{name} chứa giá trị thiếu hoặc không hữu hạn.")
    if minimum is not None and present.lt(minimum).any():
        raise ValueError(f"{name} phải lớn hơn hoặc bằng {minimum}.")
    if integer and present.mod(1).ne(0).any():
        raise ValueError(f"{name} phải là số nguyên.")
    return result
