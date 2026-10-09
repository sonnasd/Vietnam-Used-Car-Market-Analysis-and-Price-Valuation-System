import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from car_pricing.evaluation import evaluate_regression
from car_pricing.preprocessing import build_model_pipeline


def build_model_pipelines(random_state=42, min_model_frequency=10, n_jobs=4):
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(min_samples_leaf=2, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
            objective="reg:squarederror",
        ),
    }
    return {
        name: build_model_pipeline(
            estimator,
            min_model_frequency=min_model_frequency,
            dense_output=name == "XGBoost",
        )
        for name, estimator in estimators.items()
    }


def _validate_training_data(features, target, name):
    values = np.asarray(target, dtype=float)
    if values.ndim != 1 or len(features) != values.size or values.size < 2:
        raise ValueError(f"{name} cần ít nhất hai tin với đặc trưng và giá tương ứng.")
    if not np.isfinite(values).all():
        raise ValueError(f"Giá trong {name} phải hữu hạn và không được thiếu.")
    if isinstance(features, pd.DataFrame) and isinstance(target, pd.Series):
        if not features.index.equals(target.index):
            raise ValueError(f"Đặc trưng và giá trong {name} không khớp thứ tự dòng.")


def train_and_compare_models(model_pipelines, X_train, y_train, X_test, y_test, r2_target=0.85):
    _validate_training_data(X_train, y_train, "train")
    _validate_training_data(X_test, y_test, "test")
    if not model_pipelines:
        raise ValueError("Cần ít nhất một Pipeline để huấn luyện.")

    trained_models = {}
    model_predictions = {}
    rows = []
    for name, pipeline in model_pipelines.items():
        fitted_pipeline = clone(pipeline).fit(X_train, y_train)
        train_prediction = fitted_pipeline.predict(X_train)
        test_prediction = fitted_pipeline.predict(X_test)
        train_metrics = evaluate_regression(y_train, train_prediction, r2_target=r2_target)
        test_metrics = evaluate_regression(y_test, test_prediction, r2_target=r2_target)
        trained_models[name] = fitted_pipeline
        model_predictions[name] = {
            "train": pd.Series(train_prediction, index=getattr(X_train, "index", None), name="price_million"),
            "test": pd.Series(test_prediction, index=getattr(X_test, "index", None), name="price_million"),
        }
        rows.append(
            {
                "Model": name,
                **{f"{metric}_train": train_metrics[metric] for metric in ("R2", "MAE", "RMSE")},
                **{f"{metric}_test": test_metrics[metric] for metric in ("R2", "MAE", "RMSE")},
                "R2_gap": train_metrics["R2"] - test_metrics["R2"],
                "Meets_R2_target": test_metrics["Meets_R2_target"],
            }
        )

    return trained_models, model_predictions, pd.DataFrame(rows)
