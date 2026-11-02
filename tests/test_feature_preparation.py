import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from car_pricing.data import load_pricing_data
from car_pricing.feature_reports import encoded_feature_report, encoded_matrix_report
from car_pricing.features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, TEXT_FEATURES,
    build_feature_dataset, prepare_model_features,
)
from car_pricing.preprocessing import build_model_pipeline, build_preprocessor
from car_pricing.splitting import split_pricing_data


def make_sources(size=20):
    cars = pd.DataFrame({
        "list_id": np.arange(1, size + 1),
        "price_million": np.arange(100, 100 + size, dtype=float),
        "mfdate": np.full(size, 2020),
        "car_age": np.full(size, 6),
        "mileage_v2": np.arange(size) * 1000.0,
        **{column: ["Nhãn"] * size for column in CATEGORICAL_FEATURES},
    }, index=np.arange(100, 100 + size))
    flags = pd.DataFrame({
        "list_id": np.arange(1, size + 1),
        **{column: np.zeros(size, dtype=int) for column in TEXT_FEATURES},
    })
    return cars, flags


class FeatureDatasetTests(unittest.TestCase):
    def test_join_uses_normalized_keys_and_preserves_vehicle_order(self):
        cars, flags = make_sources()
        flags.loc[0, "vb_zin"] = 1
        flags["list_id"] = flags["list_id"].map(lambda value: f" {value:03d} ")
        flags = flags.sample(frac=1, random_state=9)
        before = cars.copy(deep=True)
        dataset = build_feature_dataset(cars, flags, reference_year=2026)
        pd.testing.assert_frame_equal(cars, before)
        self.assertTrue(dataset.index.equals(cars.index))
        self.assertEqual(dataset["list_id"].tolist(), cars["list_id"].tolist())
        self.assertEqual(dataset.loc[100, "vb_zin"], 1)
        self.assertEqual(dataset["vb_zin"].sum(), 1)
        self.assertEqual(tuple(dataset.columns), ("list_id", "price_million", *FEATURE_COLUMNS))

    def test_bad_keys_and_missing_or_extra_flags_fail(self):
        for change in ("duplicate", "missing", "extra", "null", "zero"):
            with self.subTest(change=change):
                cars, flags = make_sources()
                if change == "duplicate":
                    flags["list_id"] = flags["list_id"].astype(str)
                    flags.loc[1, "list_id"] = "001"
                elif change == "missing":
                    flags = flags.iloc[1:]
                elif change == "extra":
                    flags.loc[0, "list_id"] = 999
                elif change == "null":
                    flags.loc[0, "list_id"] = np.nan
                else:
                    flags.loc[0, "list_id"] = 0
                with self.assertRaises(ValueError):
                    build_feature_dataset(cars, flags)

    def test_invalid_numeric_inputs_fail(self):
        for column, value in (
            ("price_million", 0), ("price_million", np.inf),
            ("mileage_v2", -1), ("mileage_v2", np.inf), ("car_age", 1.5),
        ):
            with self.subTest(column=column, value=value):
                cars, flags = make_sources()
                cars[column] = cars[column].astype(float)
                cars.loc[100, column] = value
                with self.assertRaises(ValueError):
                    build_feature_dataset(cars, flags)

    def test_price_units_and_age_must_match_sources(self):
        cars, flags = make_sources()
        cars["price"] = cars["price_million"] * 1_000_000
        build_feature_dataset(cars, flags, reference_year=2026)
        cars.loc[100, "price"] = 1
        with self.assertRaises(ValueError):
            build_feature_dataset(cars, flags, reference_year=2026)
        cars = cars.drop(columns="price")
        for year in (2025, 2027):
            with self.subTest(reference_year=year), self.assertRaises(ValueError):
                build_feature_dataset(cars, flags, reference_year=year)
        cars.loc[100, "mfdate"] = 2027
        with self.assertRaises(ValueError):
            build_feature_dataset(cars, flags, reference_year=2026)

    def test_missing_values_remain_missing_until_pipeline(self):
        cars, flags = make_sources()
        cars["car_age"] = cars["car_age"].astype(float)
        cars.loc[100, ["mileage_v2", "car_age"]] = np.nan
        cars.loc[100, "carcolor"] = "  "
        dataset = build_feature_dataset(cars, flags)
        self.assertTrue(dataset.loc[100, ["mileage_v2", "car_age", "carcolor"]].isna().all())

    def test_nonbinary_or_missing_flags_fail(self):
        for value in (-1, 2, 0.5, np.nan):
            cars, flags = make_sources()
            flags["vb_zin"] = flags["vb_zin"].astype(float)
            flags.loc[0, "vb_zin"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_feature_dataset(cars, flags)

    def test_loader_and_duplicate_csv_headers(self):
        cars, flags = make_sources()
        with tempfile.TemporaryDirectory() as directory:
            clean_path, flags_path = Path(directory) / "cars.csv", Path(directory) / "flags.csv"
            cars.to_csv(clean_path, index=False)
            flags.to_csv(flags_path, index=False)
            loaded = load_pricing_data(clean_path, flags_path, reference_year=2026)
            self.assertEqual(len(loaded.dataset), len(cars))
            clean_path.write_text("list_id,list_id\n1,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_pricing_data(clean_path, flags_path, reference_year=2026)


class PreprocessingTests(unittest.TestCase):
    def test_test_data_cannot_change_imputation_or_categories(self):
        cars, flags = make_sources(4)
        features = prepare_model_features(build_feature_dataset(cars, flags))
        train, test = features.iloc[:3].copy(), features.iloc[3:].copy()
        train["car_age"] = [2.0, 4.0, np.nan]
        train["mileage_v2"] = [10.0, 30.0, np.nan]
        train["carmodel_name"] = ["A", "A", "B"]
        test["car_age"] = 1000.0
        test["carmodel_name"] = "Test-only model"
        preprocessor = build_preprocessor(min_model_frequency=2)
        self.assertFalse(hasattr(preprocessor, "transformers_"))
        preprocessor.fit(train)
        numeric = preprocessor.named_transformers_["numeric"]
        np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, [3, 20])
        categories = preprocessor.named_transformers_["car_model"].named_steps["encoder"].categories_[0]
        self.assertNotIn("Test-only model", categories)
        before = numeric.named_steps["scaler"].mean_.copy()
        report = encoded_matrix_report(preprocessor, train, test)
        np.testing.assert_array_equal(numeric.named_steps["scaler"].mean_, before)
        self.assertEqual(report["Số cột sau xử lý"].nunique(), 1)
        lineage = encoded_feature_report(preprocessor)
        self.assertEqual(len(lineage), preprocessor.transform(test).shape[1])
        self.assertEqual(set(lineage["Đặc trưng gốc"]), set(FEATURE_COLUMNS))
        dense = clone(build_preprocessor(min_model_frequency=2, dense_output=True)).fit(train)
        np.testing.assert_allclose(dense.transform(test), preprocessor.transform(test).toarray())

    def test_split_is_reproducible_aligned_and_excludes_target_and_id(self):
        cars, flags = make_sources()
        dataset = build_feature_dataset(cars, flags)
        first = split_pricing_data(dataset)
        second = split_pricing_data(dataset)
        self.assertEqual(len(first.X_train), 16)
        self.assertEqual(len(first.X_test), 4)
        pd.testing.assert_frame_equal(first.X_train, second.X_train)
        self.assertTrue(first.X_train.index.equals(first.y_train.index))
        self.assertTrue(first.X_test.index.equals(first.test_listing_ids.index))
        self.assertFalse(first.train_listing_ids.isin(first.test_listing_ids).any())
        self.assertEqual(tuple(first.X_train.columns), FEATURE_COLUMNS)
        pipeline = build_model_pipeline(LinearRegression()).fit(first.X_train, first.y_train)
        self.assertTrue(np.isfinite(pipeline.predict(first.X_test)).all())


if __name__ == "__main__":
    unittest.main()
