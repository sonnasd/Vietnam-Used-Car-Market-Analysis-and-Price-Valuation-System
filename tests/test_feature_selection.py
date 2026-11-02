import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from car_pricing.feature_engineering import (
    km_per_year, missing_value_report, new_feature_report, outlier_report,
)
from car_pricing.feature_selection import (
    drop_column_report, mutual_information_report, neutralize, selection_decision,
)
from car_pricing.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TEXT_FEATURES
from car_pricing.preprocessing import build_model_pipeline


def make_training_data(size=200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(0, 20, size).astype(float)
    features = pd.DataFrame({
        "car_age": age,
        "mileage_v2": rng.uniform(0, 200_000, size),
        **{column: ["Nhãn"] * size for column in CATEGORICAL_FEATURES},
        **{column: np.zeros(size, dtype=np.int8) for column in TEXT_FEATURES},
    })
    features["carbrand_name"] = rng.choice(["A", "B"], size)
    price = 800 - 30 * age + np.where(features["carbrand_name"].eq("A"), 200, 0) + rng.normal(0, 5, size)
    return features.loc[:, list(FEATURE_COLUMNS)], pd.Series(price, name="price_million")


class FeatureSelectionTests(unittest.TestCase):
    def test_neutralize_removes_information_without_changing_input(self):
        X, _ = make_training_data(10)
        before = X.copy()
        neutral = neutralize(X, ("car_age", "carbrand_name", "vb_zin"))
        pd.testing.assert_frame_equal(X, before)
        self.assertTrue(neutral["car_age"].isna().all())
        self.assertEqual(neutral["carbrand_name"].nunique(), 1)
        self.assertEqual(neutral["vb_zin"].dtype, np.int8)

    def test_informative_columns_rank_above_noise(self):
        X, y = make_training_data()
        information = mutual_information_report(X, y).set_index("Đặc trưng")["Mutual information"]
        self.assertGreater(information["car_age"], information["mileage_v2"])

        cv = KFold(n_splits=4, shuffle=True, random_state=0)
        candidates = {column: (column,) for column in ("car_age", "carbrand_name", "mileage_v2")}
        report = drop_column_report(build_model_pipeline(LinearRegression()), X, y, candidates, cv)
        decision = selection_decision(report, n_folds=4).set_index("Bỏ")["Gợi ý theo quy tắc"]
        self.assertEqual(decision["car_age"], "Giữ")
        self.assertEqual(decision["carbrand_name"], "Giữ")
        self.assertEqual(decision["mileage_v2"], "Loại")
        self.assertGreater(report.attrs["baseline_r2"], 0.95)


class FeatureEngineeringTests(unittest.TestCase):
    def test_km_per_year_avoids_division_by_zero_age(self):
        X, _ = make_training_data(3)
        X["car_age"] = [0.0, 2.0, np.nan]
        X["mileage_v2"] = [5_000.0, 40_000.0, 1_000.0]
        np.testing.assert_allclose(km_per_year(X), [5_000.0, 20_000.0, np.nan])

    def test_reports_use_train_statistics(self):
        X, y = make_training_data(20)
        X.loc[0, "car_age"] = np.nan
        X.loc[1, "carcolor"] = np.nan
        missing = missing_value_report(X).set_index("Đặc trưng")
        self.assertEqual(missing.loc["car_age", "Số thiếu"], 1)
        self.assertEqual(missing.loc["car_age", "Giá trị điền"], X["car_age"].median())
        self.assertEqual(missing.loc["carcolor", "Giá trị điền"], "Không rõ")
        outliers = outlier_report(X, y).set_index("Biến")
        self.assertIn("km_moi_nam", outliers.index)

    def test_new_feature_is_scored_on_the_same_folds(self):
        X, y = make_training_data()
        cv = KFold(n_splits=4, shuffle=True, random_state=0)
        candidates = {"tuoi_xe_lap": lambda frame: frame["car_age"].fillna(0)}
        report = new_feature_report(LinearRegression(), X, y, cv, candidates=candidates)
        self.assertEqual(report.loc[0, "Đặc trưng mới"], "tuoi_xe_lap")
        self.assertAlmostEqual(report.loc[0, "Mức tăng R²"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
