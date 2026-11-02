import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import build_feature_dataset
from .validation import normalize_listing_ids


@dataclass(frozen=True)
class PricingData:
    cars: pd.DataFrame
    text_flags: pd.DataFrame
    dataset: pd.DataFrame
    reference_year: int


def load_pricing_data(clean_path, text_flags_path, reference_year):
    sources = []
    for path in (Path(clean_path), Path(text_flags_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream), [])
        if len(columns) != len(set(columns)):
            raise ValueError(f"{path.name} có tên cột trùng nhau.")
        frame = pd.read_csv(path, dtype={"list_id": "string"}, low_memory=False)
        if "list_id" not in frame:
            raise KeyError(f"{path.name} thiếu cột list_id.")
        frame["list_id"] = normalize_listing_ids(frame["list_id"], path.name)
        sources.append(frame)
    cars, text_flags = sources
    dataset = build_feature_dataset(cars, text_flags, reference_year=reference_year)
    return PricingData(cars, text_flags, dataset, reference_year)
