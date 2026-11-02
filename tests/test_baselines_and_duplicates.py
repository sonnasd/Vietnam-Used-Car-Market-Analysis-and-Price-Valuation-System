import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from car_pricing.baselines import BASELINES, GroupMedianRegressor, baseline_comparison
from car_pricing.duplicates import cross_split_duplicates, duplicate_impact_report
from car_pricing.feature_engineering import log_target_report
from car_pricing.preprocessing import build_model_pipeline
from test_feature_selection import make_training_data


def make_cars():
    return pd.DataFrame({
        "carbrand_name": ["Toyota", "Toyota", "Toyota", "Kia", "Toyota", "Mazda"],
        "carmodel_name": ["Vios", "Vios", "Vios", "Morning", "Camry", "CX 5"],
        "car_age": [3.0, 3.0, 5.0, 4.0, 2.0, 1.0],
    })


class GroupMedianRegressorTests(unittest.TestCase):
    def test_falls_back_from_detailed_group_to_overall_median(self):
        cars = make_cars()
        prices = pd.Series([400.0, 420.0, 350.0, 250.0, 900.0, 700.0])
        levels = BASELINES["Trung vị theo hãng + dòng xe + tuổi xe"]
        baseline = GroupMedianRegressor(levels).fit(cars.iloc[:5], prices.iloc[:5])
        unseen = pd.DataFrame({
            "carbrand_name": ["Toyota", "Toyota", "Toyota", "Mazda"],
            "carmodel_name": ["Vios", "Vios", "Innova", "CX 5"],
            "car_age": [3.0, 9.0, 3.0, 1.0],
        })
        # Vios 3 năm -> nhóm dòng + tuổi; Vios 9 năm -> nhóm dòng; Innova -> nhóm hãng; Mazda -> toàn bộ.
        np.testing.assert_allclose(baseline.predict(unseen), [410.0, 400.0, 410.0, 400.0])
        self.assertEqual(baseline.predict(cars.iloc[[2, 3]]).tolist(), [350.0, 250.0])

    def test_overall_baseline_predicts_train_median(self):
        cars = make_cars()
        prices = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        baseline = GroupMedianRegressor().fit(cars, prices)
        self.assertTrue(np.all(baseline.predict(cars) == 350.0))

    def test_comparison_lists_baselines_before_models(self):
        X, y = make_training_data()
        X_train, X_test, y_train, y_test = X.iloc[:150], X.iloc[150:], y.iloc[:150], y.iloc[150:]
        model = build_model_pipeline(LinearRegression()).fit(X_train, y_train)
        report = baseline_comparison(X_train, y_train, X_test, y_test, {"Linear Regression": model})
        self.assertEqual(report["Cách dự đoán"].tolist(), [*BASELINES, "Linear Regression"])
        self.assertGreater(report.iloc[-1]["R2"], report.iloc[0]["R2"])


class DuplicateTests(unittest.TestCase):
    def test_matches_identical_listing_including_missing_labels(self):
        X, y = make_training_data(6)
        X["carcolor"] = [np.nan, "Đen", "Trắng", "Đỏ", "Bạc", "Xanh"]
        X_train, y_train = X.iloc[:3], y.iloc[:3]
        X_test = X.iloc[[0, 1, 4]].set_axis([10, 11, 12])
        y_test = pd.Series([y.iloc[0], y.iloc[1] + 5, y.iloc[4]], index=X_test.index)
        pairs = cross_split_duplicates(
            X_train, y_train, pd.Series([101, 102, 103], index=X_train.index),
            X_test, y_test, pd.Series([201, 202, 205], index=X_test.index),
        )
        self.assertEqual(pairs["list_id_train"].tolist(), [101, pd.NA, pd.NA])
        self.assertTrue(pairs.index.equals(X_test.index))

    def test_impact_report_splits_test_into_duplicate_groups(self):
        y_test = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        predictions = y_test + [0.0, 0.0, 10.0, -10.0, 20.0, -20.0]
        is_duplicate = pd.Series([True, True, False, False, False, False])
        report = duplicate_impact_report(y_test, predictions, is_duplicate).set_index("Nhóm")
        self.assertEqual(report["Số tin"].tolist(), [6, 2, 4])
        self.assertEqual(report.loc["Tin trùng với train", "MAE"], 0.0)
        self.assertEqual(report.loc["Tin không trùng", "MAE"], 15.0)


class LogTargetTests(unittest.TestCase):
    def test_reports_raw_and_log_targets_on_original_price_scale(self):
        X, y = make_training_data()
        report = log_target_report(LinearRegression(), X, y, KFold(n_splits=4, shuffle=True, random_state=0))
        self.assertEqual(report["Biến mục tiêu"].tolist(), ["Giá gốc", "log(giá)"])
        self.assertTrue(report["MAE"].between(0, 100).all())
        self.assertTrue(np.isfinite(report[["R2", "MAE", "RMSE", "MAPE (%)"]].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
