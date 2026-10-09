import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .evaluation import evaluate_regression


def build_error_table(features, y_true, y_pred, listing_ids):
    if not features.index.is_unique:
        raise ValueError("Index tập test phải duy nhất để đối chiếu từng tin.")
    for values in (y_true, y_pred, listing_ids):
        if isinstance(values, pd.Series) and not features.index.equals(values.index):
            raise ValueError("Đặc trưng, giá và mã tin không khớp thứ tự dòng.")
    evaluate_regression(y_true, y_pred)
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    identifiers = np.asarray(listing_ids)
    if len(features) != len(actual) or identifiers.ndim != 1 or len(identifiers) != len(actual):
        raise ValueError("Số tin, giá và mã tin phải có cùng độ dài.")
    if np.any(actual <= 0):
        raise ValueError("Giá thực tế phải dương để tính sai số phần trăm.")
    if pd.isna(identifiers).any() or pd.Index(identifiers).has_duplicates:
        raise ValueError("Mã tin phải duy nhất và không được thiếu.")
    table = features.copy()
    table.insert(0, "list_id", identifiers)
    table["y_true"] = actual
    table["y_pred"] = predicted
    table["residual"] = actual - predicted
    table["absolute_error"] = table["residual"].abs()
    table["percentage_error"] = table["absolute_error"] / table["y_true"] * 100
    return table


def summarize_residuals(error_table):
    if error_table.empty:
        raise ValueError("Cần ít nhất một tin để thống kê residual.")
    residual = error_table["residual"]
    absolute_error = error_table["absolute_error"]
    percentage_error = error_table["percentage_error"]
    squared_error = residual.pow(2)
    tail_size = max(1, int(np.ceil(len(error_table) * 0.05)))
    tail_share = (
        squared_error.nlargest(tail_size).sum() / squared_error.sum() * 100
        if squared_error.sum() > 0
        else 0.0
    )
    return pd.Series(
        {
            "Số tin test": len(error_table),
            "Residual trung bình (triệu VNĐ)": residual.mean(),
            "Residual trung vị (triệu VNĐ)": residual.median(),
            "MAE (triệu VNĐ)": absolute_error.mean(),
            "RMSE (triệu VNĐ)": np.sqrt(squared_error.mean()),
            "P90 sai số tuyệt đối (triệu VNĐ)": absolute_error.quantile(0.9),
            "P95 sai số tuyệt đối (triệu VNĐ)": absolute_error.quantile(0.95),
            "MAPE (%)": percentage_error.mean(),
            "Trung vị sai số phần trăm (%)": percentage_error.median(),
            "Tỷ lệ sai lệch không quá 10% (%)": percentage_error.le(10).mean() * 100,
            "Tỷ lệ sai lệch không quá 20% (%)": percentage_error.le(20).mean() * 100,
            "Tỷ lệ dự đoán cao hơn giá rao (%)": residual.lt(0).mean() * 100,
            "Tỷ lệ dự đoán thấp hơn giá rao (%)": residual.gt(0).mean() * 100,
            "Tỷ trọng SSE của 5% tin sai nhiều nhất (%)": tail_share,
        },
        name="Giá trị",
    ).to_frame()


def compute_feature_importance(model, X_test, y_test, n_repeats=5, random_state=42):
    if not X_test.index.equals(y_test.index):
        raise ValueError("Đặc trưng và giá test không khớp thứ tự dòng.")
    scores = permutation_importance(
        model,
        X_test,
        y_test,
        scoring={"R2": "r2", "MAE": "neg_mean_absolute_error"},
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "Feature": X_test.columns,
            "R2_drop": scores["R2"].importances_mean,
            "R2_drop_std": scores["R2"].importances_std,
            "MAE_increase": scores["MAE"].importances_mean,
            "MAE_increase_std": scores["MAE"].importances_std,
        }
    ).sort_values("R2_drop", ascending=False).reset_index(drop=True)


def describe_errors(error_summary, segment_report, importance_table, error_table=None):
    eligible = segment_report.loc[segment_report["eligible"]].copy()
    if eligible.empty:
        return "Chưa có phân khúc đủ số tin test để xếp hạng sai số."
    worst_absolute = eligible.sort_values("MAE", ascending=False).iloc[0]
    worst_relative = eligible.sort_values("MAPE", ascending=False).iloc[0]
    direction = (
        "thấp hơn" if worst_absolute["bias"] > 0
        else "cao hơn" if worst_absolute["bias"] < 0
        else "bằng"
    )
    lines = [
        f"- **Sai số tiền lớn nhất:** {worst_absolute['feature']} = "
        f"**{worst_absolute['segment']}**, MAE **{worst_absolute['MAE']:.2f} triệu VNĐ**, "
        f"RMSE {worst_absolute['RMSE']:.2f}; "
        f"{int(worst_absolute['n_test'])} tin test và {int(worst_absolute['n_train'])} tin train. "
        f"Giá trung vị {worst_absolute['median_actual_price']:.2f} triệu, "
        f"MAPE {worst_absolute['MAPE']:.2f}%. "
        f"Residual trung bình {worst_absolute['bias']:.2f} triệu: "
        f"dự đoán trung bình {direction} giá rao.",
        f"- **Sai số tương đối lớn nhất:** {worst_relative['feature']} = "
        f"**{worst_relative['segment']}**, MAPE **{worst_relative['MAPE']:.2f}%**, "
        f"MAE {worst_relative['MAE']:.2f} triệu trên {int(worst_relative['n_test'])} tin test; "
        f"trung vị sai số phần trăm {worst_relative['median_percentage_error']:.2f}%. "
        "MAPE dùng giá từng tin làm mẫu số nên xe giá thấp có thể sai nhiều theo phần trăm.",
    ]
    if error_table is not None and worst_relative["feature"] in error_table.columns:
        labels = error_table[worst_relative["feature"]].astype("string").str.strip()
        group_errors = error_table.loc[
            labels.fillna("Thiếu dữ liệu").eq(worst_relative["segment"])
        ]
        tail = group_errors.nlargest(2, "percentage_error")
        total_percentage_error = group_errors["percentage_error"].sum()
        if len(tail) == 2 and total_percentage_error > 0:
            share = tail["percentage_error"].sum() / total_percentage_error * 100
            if share >= 50:
                prices = ", ".join(f"{price:.2f}" for price in tail["y_true"])
                lines.append(
                    f"- **MAPE bị chi phối bởi ít tin:** hai tin sai nhiều theo phần trăm "
                    f"trong nhóm {worst_relative['segment']} có giá niêm yết {prices} triệu, "
                    f"đóng góp {share:.2f}% tổng sai số phần trăm của nhóm. "
                    "Đây là ảnh hưởng của phần đuôi và mẫu số giá thấp; "
                    "cần đối chiếu giá của hai tin gốc trước khi kết luận lỗi chung cho cả dòng xe."
                )
    price_groups = eligible.loc[eligible["feature"].eq("price_band")].sort_values(
        "median_actual_price"
    )
    if len(price_groups) >= 2:
        low, high = price_groups.iloc[0], price_groups.iloc[-1]
        lines.append(
            f"- **Ảnh hưởng của mức giá:** nhóm {high['segment']} có MAE "
            f"{high['MAE']:.2f} triệu, MAPE {high['MAPE']:.2f}%; "
            f"nhóm {low['segment']} có MAE {low['MAE']:.2f} triệu, MAPE {low['MAPE']:.2f}%. "
            "Cần đọc hai chỉ số cùng nhau: cùng một tỷ lệ lệch giá tạo ra số tiền sai lớn hơn "
            "ở xe đắt, còn MAPE nhạy với mức giá thấp."
        )
    frequency_groups = eligible.loc[eligible["feature"].eq("model_frequency_group")]
    rare = frequency_groups.loc[
        frequency_groups["segment"].str.startswith("Dòng xe hiếm", na=False)
    ]
    common = frequency_groups.loc[
        frequency_groups["segment"].str.startswith("Dòng xe từ", na=False)
    ]
    if not rare.empty and not common.empty:
        rare_row, common_row = rare.iloc[0], common.iloc[0]
        lines.append(
            f"- **Độ phủ dòng xe:** nhóm dòng xe hiếm có {int(rare_row['n_test'])} tin test, "
            f"MAE {rare_row['MAE']:.2f} triệu và MAPE {rare_row['MAPE']:.2f}%; "
            f"nhóm dòng xe phổ biến có MAE {common_row['MAE']:.2f} triệu "
            f"và MAPE {common_row['MAPE']:.2f}%. "
            "Ít mẫu và việc nhiều dòng xe dùng chung mã nhóm hiếm là khả năng cần kiểm tra; "
            "khác biệt về giá và loại xe cũng có thể ảnh hưởng kết quả này."
        )
    tail_share = error_summary.loc[
        "Tỷ trọng SSE của 5% tin sai nhiều nhất (%)", "Giá trị"
    ]
    lines.append(
        f"- **Sai số tập trung ở đuôi:** 5% tin có sai số lớn nhất đóng góp "
        f"{tail_share:.2f}% tổng sai số bình phương. "
        "Tỷ trọng này cho biết mức RMSE chịu ảnh hưởng của phần đuôi; "
        "bảng tin sai nhiều nhất dùng để kiểm tra lại."
    )
    top_features = ", ".join(importance_table.loc[
        importance_table["R2_drop"].gt(0), "Feature"
    ].head(3)) or "chưa có đặc trưng với mức giảm R² dương"
    lines.append(
        f"- **Thông tin mô hình đang dùng:** các đặc trưng đứng đầu theo mức giảm R² dương: "
        f"{top_features}. Tập đầu vào chưa có phiên bản động cơ, cấp trang bị hoặc "
        "tình trạng xe đã được kiểm chứng. Khác biệt của những yếu tố này giữa các xe cùng "
        "dòng là một giả thuyết cho các tin sai lớn, cần kiểm tra từ tin gốc."
    )
    lines.append(
        "\nCác nhận xét trên mô tả dữ liệu test và đưa ra khả năng cần kiểm tra, "
        "chưa xác nhận quan hệ nhân quả. Giá mục tiêu là giá niêm yết; chênh lệch "
        "không tự chứng minh mô hình sai về giá giao dịch hoặc người bán rao sai."
    )
    return "\n\n".join(lines)
