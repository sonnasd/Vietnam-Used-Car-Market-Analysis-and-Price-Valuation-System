import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate_regression(y_true, y_pred, r2_target=0.85):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Giá thực tế và giá dự đoán phải là hai vector có cùng độ dài.")
    if actual.size < 2:
        raise ValueError("Cần ít nhất hai tin để đánh giá R².")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Giá thực tế và giá dự đoán phải hữu hạn và không được thiếu.")
    if not np.isfinite(r2_target):
        raise ValueError("Ngưỡng R² phải hữu hạn.")

    score = float(r2_score(actual, predicted))
    return {
        "R2": score,
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(root_mean_squared_error(actual, predicted)),
        "Meets_R2_target": bool(score > r2_target),
    }
