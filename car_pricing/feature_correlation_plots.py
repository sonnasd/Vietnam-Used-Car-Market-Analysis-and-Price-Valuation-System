import matplotlib.pyplot as plt
import numpy as np

from .error_plots import BLUE, GRAY, ORANGE, _require_columns, _style_axes
from .feature_correlation import feature_group

GROUP_COLORS = {"Số": BLUE, "Danh mục": ORANGE, "Text flags": GRAY}


def plot_target_association(target_association):
    """Thanh ngang R² CV đơn biến của từng đặc trưng, thanh sai số là một độ lệch chuẩn giữa các fold."""
    _require_columns(target_association, ("Đặc trưng", "Nhóm", "R² CV đơn biến", "Độ lệch chuẩn R² CV"))
    table = target_association.sort_values("R² CV đơn biến").reset_index(drop=True)
    positions = np.arange(len(table))
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        figure, axis = plt.subplots(figsize=(10, max(4, 0.38 * len(table) + 1.5)), constrained_layout=True)
        axis.barh(
            positions, table["R² CV đơn biến"], xerr=table["Độ lệch chuẩn R² CV"].fillna(0),
            color=table["Nhóm"].map(GROUP_COLORS).fillna(GRAY), alpha=0.9, capsize=2,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(positions, table["Đặc trưng"])
        axis.set_xlabel("R² CV đơn biến (giá rao, triệu VNĐ)")
        axis.set_title("Mức giải thích giá của từng đặc trưng khi dùng riêng")
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in GROUP_COLORS.values()]
        axis.legend(handles, GROUP_COLORS.keys(), loc="lower right", frameon=False)
        _style_axes(axis)
    return figure


def plot_association_matrix(matrix, annotate_from=0.1):
    """Nửa dưới ma trận liên hệ; đường kẻ tách nhóm biến số, danh mục và cờ văn bản."""
    if list(matrix.index) != list(matrix.columns):
        raise ValueError("Ma trận liên hệ phải có cùng thứ tự hàng và cột.")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Mức liên hệ phải hữu hạn và nằm trong [0, 1].")
    size = len(matrix)
    masked = np.ma.masked_where(np.triu(np.ones((size, size), dtype=bool), k=1), values)
    groups = [feature_group(column) for column in matrix.columns]
    boundaries = [index for index in range(1, size) if groups[index] != groups[index - 1]]
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        figure, axis = plt.subplots(figsize=(0.55 * size + 3, 0.5 * size + 2), constrained_layout=True)
        image = axis.imshow(masked, cmap="Blues", vmin=0, vmax=1)
        for row in range(size):
            for column in range(row):
                value = values[row, column]
                if value >= annotate_from:
                    axis.text(
                        column, row, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value >= 0.6 else "black",
                    )
        for boundary in boundaries:
            edge = boundary - 0.5
            axis.plot([-0.5, edge], [edge, edge], color=GRAY, linewidth=1)
            axis.plot([edge, edge], [edge, size - 0.5], color=GRAY, linewidth=1)
        axis.set_xticks(range(size), matrix.columns, rotation=90)
        axis.set_yticks(range(size), matrix.index)
        axis.spines[:].set_visible(False)
        axis.set_title(
            "Mức liên hệ giữa các đặc trưng trên train\n"
            "Số–số: |Spearman ρ| · Số–danh mục: η hiệu chỉnh · Danh mục–danh mục: Cramér's V hiệu chỉnh"
        )
        figure.colorbar(image, ax=axis, shrink=0.7, label="Mức liên hệ (0–1)")
    return figure
