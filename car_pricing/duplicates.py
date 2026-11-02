import pandas as pd

from .evaluation import evaluate_regression
from .features import TARGET_COLUMN

DUPLICATE_KEYS = (
    "carbrand_name",
    "carmodel_name",
    "car_age",
    "mileage_v2",
    "gearbox",
    "fuel",
    "carcolor",
    "region_name",
)


def _signatures(features, target):
    listing = features.loc[:, list(DUPLICATE_KEYS)].assign(**{TARGET_COLUMN: target})
    return pd.util.hash_pandas_object(listing, index=False)


def cross_split_duplicates(X_train, y_train, train_ids, X_test, y_test, test_ids):
    train_matches = pd.Series(train_ids.to_numpy(), index=_signatures(X_train, y_train).to_numpy())
    train_matches = train_matches[~train_matches.index.duplicated()]
    matched_ids = _signatures(X_test, y_test).map(train_matches).astype("Int64")
    return pd.DataFrame({"list_id_test": test_ids, "list_id_train": matched_ids}, index=X_test.index)


def duplicate_impact_report(y_test, predictions, is_duplicate):
    groups = {
        "Toàn bộ test": pd.Series(True, index=y_test.index),
        "Tin trùng với train": is_duplicate,
        "Tin không trùng": ~is_duplicate,
    }
    rows = []
    for name, mask in groups.items():
        metrics = evaluate_regression(y_test[mask], predictions[mask])
        rows.append({
            "Nhóm": name,
            "Số tin": int(mask.sum()),
            "Tỷ lệ (%)": mask.mean() * 100,
            **{metric: metrics[metric] for metric in ("R2", "MAE", "RMSE")},
        })
    return pd.DataFrame(rows)
