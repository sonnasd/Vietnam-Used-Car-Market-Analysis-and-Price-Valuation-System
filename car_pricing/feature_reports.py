import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TEXT_FEATURES

FEATURE_GROUPS = {
    "Số": NUMERIC_FEATURES,
    "Danh mục": CATEGORICAL_FEATURES,
    "Text flags": TEXT_FEATURES,
}


COLUMN_ROLES = {
    "price_million": ("Mục tiêu", "Giá niêm yết, đơn vị triệu VNĐ"),
    "list_id": ("Khóa", "Ghép và đối chiếu tin; không đưa vào mô hình"),
    "mfdate": ("Nguồn tạo đặc trưng", "car_age = năm tham chiếu - mfdate"),
}
EXCLUSION_REASONS = {
    "price": "Chứa trực tiếp giá cần dự đoán (rò rỉ mục tiêu)",
    "price_string": "Chứa trực tiếp giá cần dự đoán (rò rỉ mục tiêu)",
    "ad_id": "Mã tin, không mô tả chiếc xe",
    "subject": "Văn bản thô; nội dung đã chuyển thành text flags ở notebook 04",
    "description": "Văn bản thô; nội dung đã chuyển thành text flags ở notebook 04",
    "carbrand": "Mã số của carbrand_name, trùng thông tin",
    "carmodel": "Mã số của carmodel_name, trùng thông tin",
    "region_v2": "Mã số của region_name, trùng thông tin",
    "region_name_v3": "Gộp từ region_name theo địa giới mới; ít chi tiết hơn",
    "area_name": "Cấp quận/huyện quá chi tiết, nhiều nhóm ít tin; đã dùng cấp tỉnh",
    "ward_name": "Cấp phường/xã quá chi tiết, nhiều nhóm ít tin; đã dùng cấp tỉnh",
    "list_time": "Thời điểm đăng tin, không phải thuộc tính của xe",
    "orig_list_time": "Thời điểm đăng tin, không phải thuộc tính của xe",
    "fetched_at": "Thời điểm thu thập dữ liệu, không phải thuộc tính của xe",
    "number_of_images": "Thuộc tính của tin đăng; không có khi định giá một chiếc xe",
}


def _column_role(frame, column):
    if column in FEATURE_COLUMNS:
        return "Đầu vào", "Thuộc danh sách đặc trưng; kiểm chứng ở bảng drop-column"
    if column in COLUMN_ROLES:
        return COLUMN_ROLES[column]
    if frame[column].nunique(dropna=False) == 1:
        return "Loại", "Chỉ có một giá trị trên toàn bộ dữ liệu"
    return "Loại", EXCLUSION_REASONS.get(column, "Ngoài danh sách đầu vào")


def feature_selection_report(cars, text_flags):
    rows = []
    for source, frame in (("cars_cleaned.csv", cars), ("text_flags.csv", text_flags)):
        for column in frame.columns:
            role, reason = _column_role(frame, column)
            rows.append({"Nguồn": source, "Cột": column, "Vai trò": role, "Lý do": reason})
    return pd.DataFrame(rows)


def model_frequency_report(X_train, X_test, min_frequency):
    train = X_train["carmodel_name"].fillna("Không rõ").value_counts()
    test = X_test["carmodel_name"].fillna("Không rõ").value_counts()
    labels = train.index.union(test.index, sort=False)
    report = pd.DataFrame({
        "Dòng xe": labels,
        "Số tin train": train.reindex(labels, fill_value=0).to_numpy(),
        "Số tin test": test.reindex(labels, fill_value=0).to_numpy(),
    })
    train_counts = report["Số tin train"]
    report["Nhóm theo train"] = np.select(
        [train_counts.eq(0), train_counts.lt(min_frequency)],
        ["Chưa xuất hiện trong train", "Hiếm"], default="Phổ biến",
    )
    return report.sort_values(["Số tin train", "Dòng xe"], ascending=[False, True]).reset_index(drop=True)


def encoded_feature_report(preprocessor):
    check_is_fitted(preprocessor)
    names = preprocessor.get_feature_names_out()
    rows = []
    for block, _, columns in preprocessor.transformers_:
        if block == "remainder":
            continue
        positions = preprocessor.output_indices_[block]
        for position in range(positions.start, positions.stop):
            name = names[position]
            local_name = name.split("__", 1)[1]
            column = next(
                column for column in columns
                if local_name == column or local_name.startswith(f"{column}_")
            )
            rows.append({
                "Vị trí": position,
                "Đặc trưng gốc": column,
                "Nhánh": block,
                "Cột sau xử lý": name,
            })
    return pd.DataFrame(rows)


def numeric_preprocessing_report(preprocessor):
    check_is_fitted(preprocessor)
    numeric = preprocessor.named_transformers_["numeric"]
    return pd.DataFrame({
        "Đặc trưng": NUMERIC_FEATURES,
        "Trung vị điền thiếu từ train": numeric.named_steps["imputer"].statistics_,
        "Trung bình sau điền thiếu": numeric.named_steps["scaler"].mean_,
        "Độ lệch chuẩn dùng chia": numeric.named_steps["scaler"].scale_,
    })


def encoded_matrix_report(preprocessor, X_train, X_test):
    check_is_fitted(preprocessor)
    rows = []
    for name, features in (("Train", X_train), ("Test", X_test)):
        matrix = preprocessor.transform(features)
        stored = matrix.data if hasattr(matrix, "tocsr") else np.asarray(matrix)
        if not np.isfinite(stored).all():
            raise ValueError(f"Ma trận {name} còn giá trị thiếu hoặc vô hạn sau xử lý.")
        rows.append({
            "Tập": name,
            "Số dòng": matrix.shape[0],
            "Số cột sau xử lý": matrix.shape[1],
            "Định dạng": type(matrix).__name__,
        })
    return pd.DataFrame(rows)
