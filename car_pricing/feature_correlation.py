"""Tương quan giữa đặc trưng và giá, và giữa các đặc trưng với nhau.

Mọi hàm chỉ nhận dữ liệu train. Mỗi cặp biến dùng thước đo hợp với kiểu dữ liệu:

- số – số: Spearman ρ (chỉ cần quan hệ đơn điệu, ít nhạy với giá trị lớn);
- số – danh mục/cờ: tỷ số tương quan η, hiệu chỉnh theo số nhóm;
- danh mục/cờ – danh mục/cờ: Cramér's V hiệu chỉnh độ chệch (Bergsma, 2013).

Nhãn danh mục thiếu được gom vào ``MISSING_LABEL`` và giá trị số thiếu được điền
trung vị train, giống preprocessor của mô hình.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .feature_reports import FEATURE_GROUPS
from .features import (
    CANDIDATE_FEATURES,
    CANDIDATE_TEXT_FEATURES,
    FEATURE_COLUMNS,
    MISSING_LABEL,
    NUMERIC_FEATURES,
)
from .validation import validate_columns

TARGET_SMOOTHING = 10
NUMERIC_BINS = 20
MODERATE_ASSOCIATION = 0.5
STRONG_ASSOCIATION = 0.7
WEAK_SIGNAL_R2 = 0.01
LEAKAGE_R2 = 0.95
VIF_WARNING = 5
VIF_HIGH = 10

METHOD_NUMERIC = "Spearman |ρ|"
METHOD_MIXED = "η hiệu chỉnh"
METHOD_CATEGORICAL = "Cramér's V hiệu chỉnh"


# ---------------------------------------------------------------------------
# Chuẩn bị dữ liệu
# ---------------------------------------------------------------------------

def feature_group(column):
    for group, columns in FEATURE_GROUPS.items():
        if column in columns:
            return group
    raise KeyError(f"{column} không thuộc nhóm đặc trưng nào.")


def _validate_inputs(X_train, y_train=None, columns=CANDIDATE_FEATURES):
    validate_columns(X_train, columns, "X_train")
    if y_train is None:
        return None
    if not isinstance(y_train, pd.Series) or not X_train.index.equals(y_train.index):
        raise ValueError("y_train phải là Series cùng index với X_train.")
    target = pd.to_numeric(y_train, errors="coerce").astype(float)
    if not np.isfinite(target).all() or target.nunique() < 2:
        raise ValueError("Giá train phải hữu hạn và có ít nhất hai mức khác nhau.")
    return target


def _numeric(values):
    values = pd.to_numeric(values, errors="raise").astype(float)
    return values.fillna(values.median())


def _labels(values):
    return values.astype(object).where(values.notna(), MISSING_LABEL).astype(str)


def _prepared(X_train, column):
    return _numeric(X_train[column]) if column in NUMERIC_FEATURES else _labels(X_train[column])


# ---------------------------------------------------------------------------
# Thước đo
# ---------------------------------------------------------------------------

def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denominator) if denominator > 0 else 0.0


def spearman(x, y):
    return pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def eta_squared(values, labels, adjusted=True):
    """Tỷ lệ phương sai của ``values`` giải thích bởi nhóm ``labels``.

    Bản hiệu chỉnh (epsilon², cùng dạng R² hiệu chỉnh) trừ phần η² tăng giả do số nhóm:
    biến có hàng trăm nhãn như dòng xe sẽ không được ưu ái chỉ vì chia nhỏ dữ liệu.
    """
    values = np.asarray(values, dtype=float)
    codes, uniques = pd.factorize(pd.Series(labels).to_numpy(), sort=False)
    n, k = len(values), len(uniques)
    total = float(((values - values.mean()) ** 2).sum())
    if k < 2 or total == 0:
        return 0.0
    counts = np.bincount(codes, minlength=k)
    sums = np.bincount(codes, weights=values, minlength=k)
    within = total - float((sums ** 2 / counts).sum() - values.sum() ** 2 / n)
    if not adjusted:
        return float(np.clip(1 - within / total, 0, 1))
    if n <= k:
        return np.nan
    return float(np.clip(1 - (within / (n - k)) / (total / (n - 1)), 0, 1))


def cramers_v(a, b):
    """Cramér's V hiệu chỉnh độ chệch (Bergsma, 2013), nằm trong [0, 1]."""
    table = pd.crosstab(pd.Series(a).to_numpy(), pd.Series(b).to_numpy()).to_numpy(dtype=float)
    rows, cols = table.shape
    n = table.sum()
    if rows < 2 or cols < 2 or n < 2:
        return 0.0
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    phi2 = ((table - expected) ** 2 / expected).sum() / n
    phi2 = max(0.0, phi2 - (rows - 1) * (cols - 1) / (n - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (n - 1)
    cols_corrected = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(rows_corrected - 1, cols_corrected - 1)
    return float(np.clip(np.sqrt(phi2 / denominator), 0, 1)) if denominator > 0 else 0.0


def theils_u(target, given):
    """U(target | given): tỷ lệ bất định của ``target`` mất đi khi biết ``given``.

    Bất đối xứng: U(hãng | dòng xe) = 1 nghĩa là dòng xe xác định hoàn toàn hãng.
    """
    table = pd.crosstab(pd.Series(given).to_numpy(), pd.Series(target).to_numpy()).to_numpy(dtype=float)
    joint = table / table.sum()
    target_share = joint.sum(axis=0)
    entropy = -(target_share * np.log(target_share)).sum()
    if entropy == 0:
        return 1.0
    given_share = np.broadcast_to(joint.sum(axis=1, keepdims=True), joint.shape)
    present = joint > 0
    conditional = -(joint[present] * np.log(joint[present] / given_share[present])).sum()
    return float(np.clip((entropy - conditional) / entropy, 0, 1))


def association(a, b, kind_a, kind_b):
    """Mức liên hệ không dấu trong [0, 1] và tên thước đo cho một cặp biến."""
    numeric_a, numeric_b = kind_a == "Số", kind_b == "Số"
    if numeric_a and numeric_b:
        return abs(spearman(a, b)), METHOD_NUMERIC
    if numeric_a or numeric_b:
        values, labels = (a, b) if numeric_a else (b, a)
        return float(np.sqrt(eta_squared(values, labels))), METHOD_MIXED
    return cramers_v(a, b), METHOD_CATEGORICAL


# ---------------------------------------------------------------------------
# Đặc trưng – giá
# ---------------------------------------------------------------------------

def _univariate_prediction(fit_values, fit_target, score_values, numeric, smoothing, n_bins):
    """Dự đoán giá chỉ từ một đặc trưng, học trên phần fit của fold.

    Danh mục/cờ: trung bình giá theo nhãn, co về trung bình chung với ``smoothing`` tin ảo.
    Số: chia ``n_bins`` khoảng theo phân vị của phần fit, lấy trung bình giá mỗi khoảng.
    Nhãn chưa gặp ở phần fit nhận trung bình chung.
    """
    overall = float(fit_target.mean())
    if numeric:
        edges = np.unique(np.quantile(fit_values, np.linspace(0, 1, n_bins + 1))[1:-1])
        fit_keys = np.searchsorted(edges, fit_values, side="right")
        score_keys = np.searchsorted(edges, score_values, side="right")
    else:
        fit_keys, score_keys = np.asarray(fit_values), np.asarray(score_values)
    stats = pd.Series(np.asarray(fit_target, dtype=float)).groupby(fit_keys).agg(["sum", "count"])
    encoded = (stats["sum"] + smoothing * overall) / (stats["count"] + smoothing)
    return pd.Series(score_keys).map(encoded).fillna(overall).to_numpy()


def univariate_cv_r2(values, target, folds, numeric, smoothing=TARGET_SMOOTHING, n_bins=NUMERIC_BINS):
    values, target = np.asarray(values), np.asarray(target, dtype=float)
    scores = []
    for fit_rows, score_rows in folds:
        predictions = _univariate_prediction(
            values[fit_rows], target[fit_rows], values[score_rows], numeric, smoothing, n_bins,
        )
        scores.append(r2_score(target[score_rows], predictions))
    return np.array(scores)


def target_association_report(
    X_train, y_train, cv, mutual_information=None,
    smoothing=TARGET_SMOOTHING, n_bins=NUMERIC_BINS,
):
    """Mức liên hệ của từng đặc trưng gốc với giá rao (triệu VNĐ) trên train.

    ``R² CV đơn biến`` là thước đo chung cho mọi kiểu biến: mô hình chỉ dùng một đặc trưng,
    học trên k-1 fold và chấm trên fold còn lại, nên không tăng giả theo số nhãn.
    """
    target = _validate_inputs(X_train, y_train)
    folds = list(cv.split(X_train))
    rows = []
    for column in CANDIDATE_FEATURES:
        group = feature_group(column)
        values = _prepared(X_train, column)
        row = {"Đặc trưng": column, "Nhóm": group, "Số giá trị": int(values.nunique())}
        if group in ("Số", "Text flags"):
            ordered = values if group == "Số" else X_train[column].astype(float)
            row["Pearson r"] = pearson(ordered, target)
            row["Spearman ρ"] = spearman(ordered, target)
        if group in ("Danh mục", "Text flags"):
            row["η"] = float(np.sqrt(eta_squared(target, values.to_numpy(), adjusted=False)))
            row["η² hiệu chỉnh"] = eta_squared(target, values.to_numpy())
        scores = univariate_cv_r2(values, target, folds, group == "Số", smoothing, n_bins)
        row["R² CV đơn biến"] = scores.mean()
        row["Độ lệch chuẩn R² CV"] = scores.std(ddof=1) if len(scores) > 1 else np.nan
        rows.append(row)
    columns = [
        "Đặc trưng", "Nhóm", "Số giá trị", "Pearson r", "Spearman ρ", "η", "η² hiệu chỉnh",
        "R² CV đơn biến", "Độ lệch chuẩn R² CV",
    ]
    report = pd.DataFrame(rows).reindex(columns=columns)
    if mutual_information is not None:
        validate_columns(mutual_information, ("Đặc trưng", "Mutual information"), "Bảng mutual information")
        report = report.merge(
            mutual_information.loc[:, ["Đặc trưng", "Mutual information"]],
            on="Đặc trưng", how="left", validate="one_to_one",
        )
    return report.sort_values("R² CV đơn biến", ascending=False, ignore_index=True)


def flag_price_report(X_train, y_train):
    """So sánh giá và tuổi xe giữa tin có cờ và không có cờ.

    Chênh lệch tuổi xe giúp nhận ra cờ chỉ đi kèm xe mới hơn (gây nhiễu), chứ
    không phải tự cờ làm giá cao hơn.
    """
    target = _validate_inputs(X_train, y_train)
    age = _numeric(X_train["car_age"])
    rows = []
    for column in CANDIDATE_TEXT_FEATURES:
        flagged = X_train[column].astype(int).eq(1)
        on, off = target[flagged], target[~flagged]
        price_on = on.median() if len(on) else np.nan
        price_off = off.median() if len(off) else np.nan
        rows.append({
            "Cờ": column,
            "Số tin cờ = 1": int(flagged.sum()),
            "Tỷ lệ cờ = 1 (%)": flagged.mean() * 100,
            "Giá trung vị cờ = 1": price_on,
            "Giá trung vị cờ = 0": price_off,
            "Chênh lệch giá trung vị (triệu VNĐ)": price_on - price_off,
            "Chênh lệch giá trung vị (%)": (price_on / price_off - 1) * 100 if price_off else np.nan,
            "Chênh lệch tuổi xe trung vị (năm)": (
                age[flagged].median() - age[~flagged].median() if flagged.any() and (~flagged).any() else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values(
        "Chênh lệch giá trung vị (%)", key=lambda values: values.abs(), ascending=False, ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Đặc trưng – đặc trưng
# ---------------------------------------------------------------------------

def feature_association_matrix(X_train, columns=CANDIDATE_FEATURES):
    """Ma trận đối xứng mức liên hệ không dấu trong [0, 1] giữa các đặc trưng gốc."""
    columns = list(columns)
    _validate_inputs(X_train, columns=columns)
    prepared = {column: _prepared(X_train, column).to_numpy() for column in columns}
    kinds = {column: feature_group(column) for column in columns}
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for a, b in combinations(columns, 2):
        value, _ = association(prepared[a], prepared[b], kinds[a], kinds[b])
        matrix.loc[a, b] = matrix.loc[b, a] = value
    return matrix


def association_strength(value, moderate=MODERATE_ASSOCIATION, strong=STRONG_ASSOCIATION):
    if value >= strong:
        return f"Mạnh (≥ {strong:g})"
    if value >= moderate:
        return f"Trung bình ({moderate:g}–{strong:g})"
    return f"Yếu (< {moderate:g})"


def association_pairs(
    matrix, X_train, target_association=None,
    moderate=MODERATE_ASSOCIATION, strong=STRONG_ASSOCIATION,
):
    """Các cặp đặc trưng có mức liên hệ từ ``moderate`` trở lên, mạnh nhất trước.

    Cặp danh mục có thêm Theil's U hai chiều để đọc quan hệ lồng nhau (A xác định B).
    Cặp số – số giữ dấu Spearman để biết hai biến cùng tăng hay ngược chiều.
    """
    columns = list(matrix.columns)
    _validate_inputs(X_train, columns=columns)
    univariate = (
        target_association.set_index("Đặc trưng")["R² CV đơn biến"]
        if target_association is not None else pd.Series(dtype=float)
    )
    rows = []
    for a, b in combinations(columns, 2):
        value = float(matrix.loc[a, b])
        if value < moderate:
            continue
        kind_a, kind_b = feature_group(a), feature_group(b)
        values_a, values_b = _prepared(X_train, a), _prepared(X_train, b)
        categorical = kind_a != "Số" and kind_b != "Số"
        both_numeric = kind_a == "Số" and kind_b == "Số"
        _, method = association(values_a.to_numpy(), values_b.to_numpy(), kind_a, kind_b)
        rows.append({
            "Đặc trưng A": a,
            "Đặc trưng B": b,
            "Thước đo": method,
            "Mức liên hệ": value,
            "Mức độ": association_strength(value, moderate, strong),
            "Spearman ρ có dấu": spearman(values_a, values_b) if both_numeric else np.nan,
            "Theil U (A | B)": theils_u(values_a, values_b) if categorical else np.nan,
            "Theil U (B | A)": theils_u(values_b, values_a) if categorical else np.nan,
            "R² CV đơn biến A": univariate.get(a, np.nan),
            "R² CV đơn biến B": univariate.get(b, np.nan),
        })
    columns_out = [
        "Đặc trưng A", "Đặc trưng B", "Thước đo", "Mức liên hệ", "Mức độ", "Spearman ρ có dấu",
        "Theil U (A | B)", "Theil U (B | A)", "R² CV đơn biến A", "R² CV đơn biến B",
    ]
    return pd.DataFrame(rows, columns=columns_out).sort_values(
        "Mức liên hệ", ascending=False, ignore_index=True,
    )


def vif_report(X_train, columns=(*NUMERIC_FEATURES, *CANDIDATE_TEXT_FEATURES)):
    """Variance inflation factor của các đặc trưng vào mô hình thành đúng một cột.

    Biến danh mục bị loại vì sau one-hot các cột của cùng một biến luôn cộng lại bằng 1,
    VIF khi đó vô hạn theo cấu trúc chứ không do dữ liệu.
    """
    columns = list(columns)
    _validate_inputs(X_train, columns=columns)
    data = pd.DataFrame({column: _numeric(X_train[column]) for column in columns})
    constant = data.columns[data.std(ddof=0).eq(0)]
    data = data.drop(columns=constant)
    data = (data - data.mean()) / data.std(ddof=0)
    rows = []
    for column in data.columns:
        others = data.drop(columns=column).to_numpy()
        design = np.column_stack([np.ones(len(data)), others])
        target = data[column].to_numpy()
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        r2 = 1 - (residual @ residual) / (target @ target)
        vif = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
        rows.append({
            "Đặc trưng": column,
            "R² theo các cột còn lại": r2,
            "VIF": vif,
            "Đánh giá": (
                f"Cao (≥ {VIF_HIGH})" if vif >= VIF_HIGH
                else f"Cần chú ý ({VIF_WARNING}–{VIF_HIGH})" if vif >= VIF_WARNING
                else f"Thấp (< {VIF_WARNING})"
            ),
        })
    rows += [
        {"Đặc trưng": column, "R² theo các cột còn lại": np.nan, "VIF": np.nan, "Đánh giá": "Hằng số trên train"}
        for column in constant
    ]
    return pd.DataFrame(rows).sort_values("VIF", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Tổng hợp cho bước huấn luyện
# ---------------------------------------------------------------------------

def correlation_summary(
    target_association, matrix, vif=None, drop_column=None, selected=FEATURE_COLUMNS, strong=STRONG_ASSOCIATION,
):
    """Một dòng mỗi đặc trưng: tín hiệu với giá, đặc trưng liên hệ mạnh nhất, VIF,
    kết quả drop-column CV (nếu có) và cảnh báo ngắn."""
    others = matrix.where(~np.eye(len(matrix), dtype=bool))
    partner = others.idxmax(axis=1)
    strength = others.max(axis=1)
    vif_values = vif.set_index("Đặc trưng")["VIF"] if vif is not None else pd.Series(dtype=float)
    drops = drop_column.set_index("Bỏ") if drop_column is not None else pd.DataFrame()
    rows = []
    for _, row in target_association.iterrows():
        column = row["Đặc trưng"]
        score = row["R² CV đơn biến"]
        warnings = []
        if score >= LEAKAGE_R2:
            warnings.append("Nghi rò rỉ mục tiêu")
        if strength[column] >= strong:
            warnings.append("Liên hệ mạnh")
        if score < WEAK_SIGNAL_R2:
            warnings.append("Tín hiệu đơn biến yếu")
        if vif_values.get(column, 0) >= VIF_WARNING:
            warnings.append("VIF cao")
        summary = {
            "Đặc trưng": column,
            "Đưa vào mô hình": "Có" if column in selected else "Không",
            "R² CV đơn biến": score,
            "Liên hệ mạnh nhất với": partner[column],
            "Mức liên hệ": strength[column],
            "VIF": vif_values.get(column, np.nan),
        }
        if drop_column is not None:
            found = column in drops.index
            summary["Mức giảm R² khi bỏ"] = drops.at[column, "Mức giảm R²"] if found else np.nan
            summary["Gợi ý drop-column"] = (
                drops.at[column, "Gợi ý theo quy tắc"]
                if found and "Gợi ý theo quy tắc" in drops.columns else "-"
            )
        summary["Cảnh báo"] = ", ".join(warnings) if warnings else "-"
        rows.append(summary)
    return pd.DataFrame(rows)


def _names(columns):
    return ", ".join(f"`{column}`" for column in columns) if len(columns) else "không có"


def describe_correlations(
    target_association, pairs, vif=None, flag_report=None, drop_column=None, strong=STRONG_ASSOCIATION,
):
    """Nhận xét Markdown ngắn sinh từ số liệu, để chữ luôn khớp lần chạy hiện tại."""
    ranked = target_association.sort_values("R² CV đơn biến", ascending=False)
    indexed = ranked.set_index("Đặc trưng")
    top = ", ".join(f"`{name}` ({score:.2f})" for name, score in ranked.head(3)[["Đặc trưng", "R² CV đơn biến"]].to_numpy())
    numeric = ", ".join(f"`{name}` ρ = {indexed.at[name, 'Spearman ρ']:.2f}" for name in NUMERIC_FEATURES)
    weak = ranked.loc[ranked["R² CV đơn biến"] < WEAK_SIGNAL_R2, "Đặc trưng"].tolist()
    lines = [
        f"- **Tín hiệu mạnh nhất (R² CV đơn biến):** {top}. Tương quan Spearman với giá: {numeric}.",
        f"- **Tín hiệu yếu khi đứng riêng (R² CV < {WEAK_SIGNAL_R2:g}):** {_names(weak)}.",
    ]

    strong_pairs = pairs.loc[pairs["Mức liên hệ"] >= strong]
    details = []
    for _, pair in strong_pairs.iterrows():
        a, b = pair["Đặc trưng A"], pair["Đặc trưng B"]
        detail = f"`{a}`–`{b}` {pair['Mức liên hệ']:.2f}"
        u_ab, u_ba = pair["Theil U (A | B)"], pair["Theil U (B | A)"]
        if pd.notna(u_ab) and max(u_ab, u_ba) >= 0.9:
            given, target, value = (b, a, u_ab) if u_ab >= u_ba else (a, b, u_ba)
            detail += f" (`{given}` gần như xác định `{target}`, U = {value:.2f})"
        details.append(detail)
    lines.append(f"- **Cặp liên hệ mạnh (≥ {strong:g}):** " + ("; ".join(details) if details else "không có") + ".")

    if flag_report is not None:
        clear = flag_report.loc[flag_report["Chênh lệch giá trung vị (%)"].abs() >= 10]
        younger = clear.loc[clear["Chênh lệch tuổi xe trung vị (năm)"] < 0, "Chênh lệch tuổi xe trung vị (năm)"].abs()
        if len(clear):
            line = f"- **Cờ văn bản:** {len(clear)}/{len(flag_report)} cờ chênh giá trung vị từ 10% trở lên"
            if len(younger):
                span = f"{younger.min():g}" if younger.min() == younger.max() else f"{younger.min():g}–{younger.max():g}"
                line += f", {len(younger)} cờ trong đó đi kèm xe trẻ hơn {span} năm, nên chênh lệch một phần do tuổi xe"
            lines.append(line + ".")

    if drop_column is not None and not drop_column.empty:
        top_drop = drop_column.sort_values("Mức giảm R²", ascending=False).iloc[0]
        noise = drop_column["Mức giảm R²"].abs() < drop_column["Độ lệch chuẩn mức giảm"]
        baseline = drop_column.attrs.get("baseline_r2")
        line = "- **Drop-column CV:** "
        if baseline is not None:
            line += f"R² CV đủ đặc trưng {baseline:.4f}; "
        line += (
            f"bỏ `{top_drop['Bỏ']}` giảm nhiều nhất ({top_drop['Mức giảm R²']:.4f}); "
            f"{int(noise.sum())}/{len(drop_column)} phương án nằm trong dao động giữa các fold"
        )
        if "Gợi ý theo quy tắc" in drop_column.columns:
            suggested = drop_column.loc[drop_column["Gợi ý theo quy tắc"].eq("Loại"), "Bỏ"].tolist()
            single = [name for name in suggested if name in CANDIDATE_FEATURES]
            groups = [name for name in suggested if name not in CANDIDATE_FEATURES]
            line += f"; quy tắc chọn loại: {_names(single)}"
            if groups:
                line += (
                    f" (phương án bỏ cả nhóm {_names(groups)} cũng dưới ngưỡng nhưng không áp dụng, "
                    "vì nhóm gồm cả cờ được giữ)"
                )
        lines.append(line + ".")

    leakage = ranked.loc[ranked["R² CV đơn biến"] >= LEAKAGE_R2, "Đặc trưng"].tolist()
    check = []
    if vif is not None and vif["VIF"].notna().any():
        worst = vif.loc[vif["VIF"].idxmax()]
        check.append(f"VIF lớn nhất {worst['VIF']:.2f} (`{worst['Đặc trưng']}`)")
    check.append(
        f"nghi rò rỉ: {_names(leakage)}" if leakage
        else f"không đặc trưng nào một mình đạt R² CV ≥ {LEAKAGE_R2:g}, không có dấu hiệu rò rỉ"
    )
    lines.append("- **Đa cộng tuyến và rò rỉ:** " + "; ".join(check) + ".")
    return "\n".join(lines)
