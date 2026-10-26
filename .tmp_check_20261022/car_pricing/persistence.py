import hashlib
import os
import pickle
import tempfile
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .features import FEATURE_COLUMNS


def _validate_pipeline(pipeline):
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Phải lưu toàn bộ sklearn Pipeline.")
    if tuple(pipeline.named_steps) != ("preprocessor", "model"):
        raise ValueError("Pipeline phải gồm hai bước preprocessor và model.")
    check_is_fitted(pipeline.named_steps["preprocessor"])
    check_is_fitted(pipeline.named_steps["model"])
    if tuple(pipeline.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("Pipeline không khớp bộ đặc trưng đầu vào của dự án.")


def _predict(pipeline, features):
    predictions = np.asarray(pipeline.predict(features), dtype=float)
    if predictions.shape != (len(features),) or not np.isfinite(predictions).all():
        raise ValueError("Dự đoán phải có một giá trị hữu hạn cho mỗi dòng dữ liệu.")
    return predictions


def _file_sha256(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_pipeline(pipeline, model_path, validation_features, max_bytes=100_000_000):
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral) or max_bytes <= 0:
        raise ValueError("Giới hạn dung lượng phải là số nguyên dương.")
    _validate_pipeline(pipeline)
    if not isinstance(validation_features, pd.DataFrame) or validation_features.empty:
        raise ValueError("Dữ liệu kiểm tra phải là DataFrame có ít nhất một dòng.")
    if tuple(validation_features.columns) != FEATURE_COLUMNS:
        raise ValueError("Dữ liệu kiểm tra phải đủ các đặc trưng thô theo đúng thứ tự.")
    original_predictions = _predict(pipeline, validation_features)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{model_path.name}.", suffix=".tmp", dir=model_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(pipeline, temporary_path, compress=3, protocol=pickle.HIGHEST_PROTOCOL)
        file_size = temporary_path.stat().st_size
        if not 0 < file_size < max_bytes:
            raise ValueError(
                f"File mô hình có {file_size:,} byte; phải nhỏ hơn {max_bytes:,} byte."
            )
        loaded_pipeline = joblib.load(temporary_path)
        _validate_pipeline(loaded_pipeline)
        loaded_predictions = _predict(loaded_pipeline, validation_features)
        np.testing.assert_allclose(
            loaded_predictions, original_predictions, rtol=1e-10, atol=1e-10
        )
        save_info = {
            "file_size_bytes": file_size,
            "file_size_mb": file_size / 1_000_000,
            "sha256": _file_sha256(temporary_path),
            "verified_rows": len(validation_features),
            "n_features": len(FEATURE_COLUMNS),
            "max_prediction_difference": float(
                np.max(np.abs(loaded_predictions - original_predictions))
            ),
        }
        os.replace(temporary_path, model_path)
        return loaded_pipeline, save_info
    finally:
        temporary_path.unlink(missing_ok=True)
