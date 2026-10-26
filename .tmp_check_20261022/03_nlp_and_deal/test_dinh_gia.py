import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from dinh_gia_form import FormDinhGia, chuan_bi_xe, goi_y_thong_so


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / "03_nlp_and_deal")
sys.path.insert(0, str(ROOT))
data = pd.read_csv(ROOT / "data/processed/cars_cleaned.csv", low_memory=False)
model = joblib.load(ROOT / "models/car_pricing_model.pkl")
features = list(model.feature_names_in_)
files = [p for folder in ("data", "models") for p in (ROOT / folder).rglob("*") if p.is_file()]
hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
namespace = {}
notebook = json.loads(Path("06_phat_hien_deal.ipynb").read_text(encoding="utf-8"))
for i in (1, 5, 7, 23):
    exec(compile("".join(notebook["cells"][i]["source"]), f"notebook06:{i}", "exec"), namespace)
predict = namespace["dinh_gia_xe"]
for value in (-15, 15):
    assert namespace["gan_nhan"](value) == "Giá hợp lý"
assert namespace["gan_nhan"](-15.01) == "Deal hời / Rẻ bất thường"
assert namespace["gan_nhan"](15.01) == "Hét giá cao"

args = ["Toyota", "Vios", 2020, 60000, "Số tự động", "Hà Nội"]
assert "Nhãn" not in predict(*args)
sample = pd.read_csv(ROOT / "models/error_analysis/test_errors.csv").iloc[0]
extra = {c: sample[c] for c in features if c.startswith("vb_") or c in ("co_mo_ta", "fuel", "cartype", "carcolor", "carorigin", "carseats")}
assert predict(sample.carbrand_name, sample.carmodel_name, 2026 - sample.car_age,
               sample.mileage_v2, sample.gearbox, sample.region_name,
               thong_tin_them=extra)["Giá dự đoán (triệu)"] == round(sample.y_pred, 1)
assert "Nhãn" in predict(*args, gia_rao=420, thong_tin_them={"vb_chinh_chu": 1, "co_mo_ta": 0})
for position, value in [(0, "BMW"), (2, 2027), (2, 2020.5), (3, -1), (3, float("nan")), (3, 2_000_001), (4, "Không rõ")]:
    invalid = args.copy()
    invalid[position] = value
    try:
        predict(*invalid)
    except ValueError:
        pass
    else:
        raise AssertionError((position, value))
for value in (0, -1, float("inf"), "abc"):
    try:
        predict(*args, gia_rao=value)
    except ValueError:
        pass
    else:
        raise AssertionError(value)
for value in (float("nan"), float("inf"), -1):
    class InvalidModel:
        def predict(self, _):
            return [value]
    namespace["model"] = InvalidModel()
    assert predict(*args) == {"Kết quả": "Không định giá được"}
namespace["model"] = model

form = FormDinhGia(data, predict)
form.hang.value = "Toyota"
form.dong.value = "Vios"
form.truong["region_name"].value = "Hà Nội"
form.odo.value = "60000"
form.co["vb_chinh_chu"].value = True
form.ket_qua.value = "old result"
form.hang.value = "BMW"
assert form.dong.value is None and "Vios" not in [v for _, v in form.dong.options]
assert form.odo.value == "" and not form.co["vb_chinh_chu"].value
assert form.ket_qua.value == "" and form.truong["region_name"].value == "Hà Nội"
for brand, name in [("BMW", "M3"), ("Haima", "M3"), ("Daewoo", "Matiz"), ("Chevrolet", "Matiz"), ("Toyota", "Dòng khác")]:
    form.hang.value = brand
    form.dong.value = name
    frame, _ = chuan_bi_xe(data, features, 2026, brand, name, 2020, 10000, "Số tự động", "Hà Nội")
    assert frame.iloc[0].carbrand_name == brand
form.hang.value = "Toyota"
form.dong.value = "Vios"
form.nam.value = 2020
assert all(form.truong[c].value == "Không rõ" for c in form.goi_y)
form.ap_dung_goi_y()
assert all(form.truong[c].value == v for c, (v, _) in form.goi_y.items())
form.gia.value = "420"
form.odo.value = "60000"
form.nam.value = 1980
assert not form.goi_y and form.nut_goi_y.disabled
assert form.gia.value == "" and form.odo.value == ""
assert "chưa có mẫu" in form.thong_bao.value
assert goi_y_thong_so(data.iloc[:1], data.iloc[0].carbrand_name, data.iloc[0].carmodel_name, data.iloc[0].mfdate)[1] == {}
fixture = pd.concat([data.loc[data.carbrand_name.eq("Toyota") & data.carmodel_name.eq("Vios")].head(1)] * 10, ignore_index=True)
fixture["mfdate"] = 2020
fixture["gearbox"] = ["Số tự động"] * 8 + ["Số sàn"] * 2
assert goi_y_thong_so(fixture, "Toyota", "Vios", 2020)[1]["gearbox"] == ("Số tự động", 0.8)
fixture.loc[0, "gearbox"] = "Số sàn"
assert "gearbox" not in goi_y_thong_so(fixture, "Toyota", "Vios", 2020)[1]

test = pd.read_csv(ROOT / "models/error_analysis/test_errors.csv")
np.testing.assert_allclose(model.predict(test[features]), test.y_pred, rtol=1e-6, atol=1e-5)
valid = test[test.y_pred > 0]
delta = (valid.y_true - valid.y_pred) / valid.y_pred * 100
assert ((delta < -15).sum(), delta.between(-15, 15).sum(), (delta > 15).sum()) == (319, 2040, 344)
assert test.y_pred.le(0).sum() == 1
assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
print("PASS: validation, widget transitions, suggestions, Deal boundaries, 2704 predictions, unchanged data/models")
