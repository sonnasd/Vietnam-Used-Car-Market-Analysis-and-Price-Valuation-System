import matplotlib.pyplot as plt
import numpy as np

FEATURE_LABELS = {
    "carbrand_name": "Hãng xe",
    "carmodel_name": "Dòng xe",
    "cartype": "Kiểu dáng",
    "gearbox": "Hộp số",
    "fuel": "Nhiên liệu",
    "region_name": "Khu vực",
    "car_age": "Tuổi xe",
    "mileage_v2": "Số km đã đi",
    "carseats": "Số chỗ ngồi",
    "carorigin": "Xuất xứ",
    "carcolor": "Màu sắc",
    "vb_bao_test": "Cờ bao test",
    "co_mo_ta": "Có mô tả",
    "car_age_group": "Tuổi xe",
    "price_band": "Khoảng giá",
    "mileage_group": "Khoảng số km",
    "model_frequency_group": "Tần suất dòng xe",
}
BLUE = "#35688A"
ORANGE = "#B46B3D"
GRAY = "#64748B"


def _require_columns(table, columns):
    missing = set(columns).difference(table.columns)
    if missing:
        raise KeyError(f"Thiếu cột biểu đồ: {', '.join(sorted(missing))}.")


def _top_rows(table, column, top_n):
    if not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("Số hàng hiển thị phải là số nguyên dương.")
    return table.sort_values(column, ascending=False).head(top_n).iloc[::-1]


def _style_axes(axes):
    for axis in np.asarray(axes, dtype=object).flat:
        axis.set_axisbelow(True)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)


def plot_residuals(error_table, model_name):
    columns = ("y_true", "y_pred", "residual", "absolute_error")
    _require_columns(error_table, columns)
    values = error_table.loc[:, list(columns)].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bảng sai số phải có dữ liệu hữu hạn để vẽ biểu đồ.")
    actual, predicted, residual, absolute_error = values.T
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        figure.suptitle(f"Phân tích sai số trên tập test: {model_name}", fontsize=15)
        axes[0, 0].scatter(actual, predicted, color=BLUE, alpha=0.45, s=15, rasterized=True)
        lower, upper = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        padding = max((upper - lower) * 0.04, 1.0)
        limits = (lower - padding, upper + padding)
        axes[0, 0].plot(limits, limits, color=ORANGE, linestyle="--", label="Dự đoán đúng")
        axes[0, 0].set(xlim=limits, ylim=limits, title="Giá thực tế và giá dự đoán",
                       xlabel="Giá thực tế (triệu đồng)", ylabel="Giá dự đoán (triệu đồng)")
        axes[0, 0].legend(frameon=False)
        axes[0, 1].scatter(predicted, residual, color=BLUE, alpha=0.45, s=15, rasterized=True)
        axes[0, 1].axhline(0, color=ORANGE, linestyle="--")
        axes[0, 1].set(title="Residual = giá thực tế − giá dự đoán",
                       xlabel="Giá dự đoán (triệu đồng)", ylabel="Residual (triệu đồng)")
        axes[1, 0].hist(residual, bins=40, color=BLUE, alpha=0.85, edgecolor="white")
        axes[1, 0].axvline(0, color=GRAY, linestyle="--", label="Residual = 0")
        axes[1, 0].axvline(residual.mean(), color=ORANGE,
                           label=f"Trung bình: {residual.mean():.2f} triệu")
        axes[1, 0].set(title="Phân bố residual", xlabel="Residual (triệu đồng)", ylabel="Số tin test")
        axes[1, 0].legend(frameon=False)
        axes[1, 1].scatter(actual, absolute_error, color=BLUE, alpha=0.45, s=15, rasterized=True)
        axes[1, 1].set(title="Sai số tuyệt đối theo giá thực tế",
                       xlabel="Giá thực tế (triệu đồng)", ylabel="Sai số tuyệt đối (triệu đồng)")
        _style_axes(axes)
    return figure


def plot_feature_importance(importance_table, top_n=12):
    _require_columns(importance_table, ("Feature", "R2_drop", "R2_drop_std"))
    rows = _top_rows(importance_table, "R2_drop", top_n)
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        figure, axis = plt.subplots(figsize=(11, max(4, 0.48 * len(rows) + 1.6)), constrained_layout=True)
        labels = rows["Feature"].map(lambda feature: FEATURE_LABELS.get(feature, feature))
        axis.barh(labels, rows["R2_drop"], xerr=rows["R2_drop_std"], color=BLUE,
                  error_kw={"ecolor": GRAY, "capsize": 3, "linewidth": 1})
        axis.axvline(0, color=GRAY, linewidth=1)
        axis.set(title="Permutation importance trên tập test",
                 xlabel="Mức giảm R² khi xáo trộn đặc trưng (±1 độ lệch chuẩn)")
        _style_axes([axis])
    return figure


def plot_segment_errors(segment_report, top_n=12):
    _require_columns(segment_report, ("feature", "segment", "n_test", "MAE", "eligible"))
    eligible = segment_report.loc[segment_report["eligible"].eq(True) & segment_report["n_test"].gt(0)]
    rows = _top_rows(eligible, "MAE", top_n)
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        figure, axis = plt.subplots(figsize=(12, max(4, 0.5 * len(rows) + 1.8)), constrained_layout=True)
        if rows.empty:
            axis.text(0.5, 0.5, "Chưa có phân khúc đủ số tin test để xếp hạng.",
                      ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            return figure
        labels = [
            f"{FEATURE_LABELS.get(row.feature, row.feature)}: {row.segment} (n={int(row.n_test)})"
            for row in rows.itertuples()
        ]
        bars = axis.barh(labels, rows["MAE"], color=BLUE)
        axis.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        axis.set_xlim(0, max(float(rows["MAE"].max()) * 1.17, 1))
        axis.set(title="Phân khúc có MAE cao nhất trong các nhóm đủ mẫu",
                 xlabel="MAE trên tập test (triệu đồng)")
        _style_axes([axis])
    return figure
