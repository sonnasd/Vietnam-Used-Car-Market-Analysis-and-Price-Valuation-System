import math
from html import escape

import ipywidgets as widgets
import pandas as pd

from phan_tich_mo_ta import TEN_CO, tom_tat_mo_ta

TRUONG_GOI_Y = {
    "gearbox": "Hộp số", "fuel": "Nhiên liệu",
    "cartype": "Kiểu xe", "carseats": "Số chỗ",
}


def so_hop_le(value, ten, minimum, maximum=None, nguyen=False):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{ten} phải là số hợp lệ.") from exc
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{ten} phải là số hữu hạn.")
    if number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{ten} nằm ngoài giới hạn cho phép.")
    if nguyen and not number.is_integer():
        raise ValueError(f"{ten} phải là số nguyên.")
    return int(number) if nguyen else number


def chuan_bi_xe(data, features, nam_du_lieu, hang, dong_xe, nam_san_xuat,
                so_km, hop_so, tinh, gia_rao=None, thong_tin_them=None):
    nhom = data[(data.carbrand_name == hang) & (data.carmodel_name == dong_xe)]
    if nhom.empty:
        raise ValueError("Cần chọn hãng và dòng xe tương ứng trong dữ liệu.")
    nam = so_hop_le(nam_san_xuat, "Năm sản xuất", 1980, nam_du_lieu, nguyen=True)
    odo = so_hop_le(so_km, "ODO", 0, 2_000_000, nguyen=True)
    gia = None
    if gia_rao is not None:
        gia = so_hop_le(gia_rao, "Giá rao (triệu đồng)", 0)
        if gia == 0:
            raise ValueError("Giá rao phải lớn hơn 0.")
    xe = {cot: 0 if cot == "co_mo_ta" or cot.startswith("vb_") else "Không rõ"
          for cot in features}
    them = {} if thong_tin_them is None else dict(thong_tin_them)
    cho_phep = set(features) - {
        "carbrand_name", "carmodel_name", "car_age", "mileage_v2", "gearbox", "region_name"
    }
    if set(them) - cho_phep:
        raise ValueError("Thông tin thêm chứa cột không được phép hoặc ghi đè thông số chính.")
    xe.update(them)
    xe.update(carbrand_name=hang, carmodel_name=dong_xe, car_age=nam_du_lieu - nam,
              mileage_v2=odo, gearbox=hop_so, region_name=tinh)
    for cot, value in xe.items():
        if cot == "co_mo_ta" or cot.startswith("vb_"):
            if value not in (0, 1):
                raise ValueError(f"{cot} chỉ nhận 0 hoặc 1.")
        elif cot not in ("car_age", "mileage_v2"):
            if pd.isna(value) and cot not in ("gearbox", "region_name", "carbrand_name", "carmodel_name"):
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{cot} không hợp lệ.")
            if value == "Không rõ" and cot not in ("gearbox", "region_name"):
                continue
            if value not in set(data[cot].dropna()):
                raise ValueError(f"{cot} không thuộc danh sách lựa chọn.")
    return pd.DataFrame([xe], columns=features), gia


def goi_y_thong_so(data, hang, dong, nam):
    nhom = data[(data.carbrand_name == hang) & (data.carmodel_name == dong)
                & (data.mfdate == nam)]
    ket_qua = {}
    if len(nhom) >= 10:
        for cot in TRUONG_GOI_Y:
            dem = nhom[cot].dropna().value_counts()
            if not dem.empty and dem.iloc[0] / len(nhom) >= 0.8:
                ket_qua[cot] = (dem.index[0], float(dem.iloc[0] / len(nhom)))
    return len(nhom), ket_qua


class FormDinhGia:
    def __init__(self, data, dinh_gia, nam_du_lieu=2026):
        self.data = data
        self.dinh_gia = dinh_gia
        self.dang_cap_nhat = False
        self.goi_y = {}
        self.hang = widgets.Dropdown(options=[("Chọn hãng", None), *((v, v) for v in sorted(data.carbrand_name.dropna().unique()))], description="Hãng:")
        self.dong = widgets.Dropdown(options=[("Chọn dòng", None)], description="Dòng xe:")
        self.nam = widgets.IntText(value=2020, description="Năm SX:")
        self.odo = widgets.Text(description="ODO (km):", placeholder="Nhập số km")
        self.gia = widgets.Text(description="Giá rao:", placeholder="Triệu đồng; có thể để trống")
        nhan = {**TRUONG_GOI_Y, "region_name": "Tỉnh/thành", "carcolor": "Màu xe", "carorigin": "Xuất xứ"}
        self.truong = {
            cot: widgets.Dropdown(options=["Không rõ", *sorted(data[cot].dropna().unique())], description=ten + ":")
            for cot, ten in nhan.items()
        }
        self.co_mo_ta = widgets.Checkbox(description="Tin có mô tả chi tiết", indent=False)
        self.co = {cot: widgets.Checkbox(description=ten, indent=False) for cot, ten in TEN_CO.items()}
        self.thong_bao = widgets.HTML()
        self.ket_qua = widgets.HTML()
        self.nut_goi_y = widgets.Button(description="Áp dụng gợi ý", disabled=True)
        self.nut = widgets.Button(description="Định giá", button_style="primary")
        self.hang.observe(self.doi_hang, names="value")
        self.dong.observe(self.doi_xe, names="value")
        self.nam.observe(self.doi_xe, names="value")
        self.nut_goi_y.on_click(self.ap_dung_goi_y)
        self.nut.on_click(self.hien_ket_qua)
        for o in [self.odo, self.gia, self.co_mo_ta, *self.truong.values(), *self.co.values()]:
            o.observe(self.xoa_ket_qua, names="value")
        trai = widgets.VBox([self.hang, self.dong, self.nam, self.odo, self.truong["gearbox"], self.truong["region_name"]])
        phai = widgets.VBox([self.truong[c] for c in ["fuel", "cartype", "carseats", "carcolor", "carorigin"]] + [self.gia])
        for o in [self.hang, self.dong, self.nam, self.odo, self.gia, *self.truong.values()]:
            o.layout.width = "calc(100% - 8px)"
            o.layout.min_width = "0"
            o.style.description_width = "90px"
        self.view = widgets.VBox([
            widgets.HTML("<h3>Thông số xe</h3>"),
            widgets.GridBox([trai, phai], layout=widgets.Layout(grid_template_columns="repeat(2, minmax(300px, 1fr))", grid_gap="24px")),
            self.thong_bao, self.nut_goi_y,
            widgets.HTML("<h3>Thông tin người bán công bố</h3><p>Có thể lấy từ tiêu đề hoặc mô tả; không phải tình trạng đã kiểm chứng.</p>"),
            self.co_mo_ta,
            widgets.GridBox(list(self.co.values()), layout=widgets.Layout(grid_template_columns="repeat(2, minmax(300px, 1fr))")),
            self.nut, self.ket_qua,
        ], layout=widgets.Layout(max_width="1050px", width="100%"))
        self.cap_nhat_goi_y()

    def xoa_ket_qua(self, _=None):
        self.ket_qua.value = ""

    def doi_hang(self, _):
        self.dang_cap_nhat = True
        self.dong.options = [("Chọn dòng", None), *((v, v) for v in sorted(self.data.loc[self.data.carbrand_name == self.hang.value, "carmodel_name"].dropna().unique()))]
        self.dong.value = None
        self.dang_cap_nhat = False
        self.doi_xe()

    def doi_xe(self, _=None):
        if self.dang_cap_nhat:
            return
        for cot, o in self.truong.items():
            if cot != "region_name":
                o.value = "Không rõ"
        self.odo.value = ""
        self.gia.value = ""
        self.co_mo_ta.value = False
        for o in self.co.values():
            o.value = False
        self.xoa_ket_qua()
        self.cap_nhat_goi_y()

    def cap_nhat_goi_y(self):
        self.goi_y = {}
        nhom = self.data[(self.data.carbrand_name == self.hang.value) & (self.data.carmodel_name == self.dong.value)]
        nam = sorted(nhom.mfdate.dropna().astype(int).unique())
        if nhom.empty:
            text = "Chọn hãng và dòng xe để xem gợi ý."
        else:
            so_tin, self.goi_y = goi_y_thong_so(self.data, self.hang.value, self.dong.value, self.nam.value)
            text = "Năm có trong dữ liệu: " + ", ".join(map(str, nam)) + f". Nhóm năm đang chọn: {so_tin} tin."
            if self.nam.value not in nam:
                text += " Năm này chưa có mẫu; không đồng nghĩa xe không tồn tại."
            if self.goi_y:
                text += "<br>Gợi ý: " + "; ".join(f"{TRUONG_GOI_Y[c]}: {escape(str(v))} ({p:.0%})" for c, (v, p) in self.goi_y.items())
            else:
                text += " Chưa đủ 10 tin hoặc chưa đạt tỷ lệ 80% để gợi ý."
        self.thong_bao.value = "<p>" + text + "</p><p>Gợi ý từ tin rao, không phải cấu hình kỹ thuật chuẩn. Bấm áp dụng rồi kiểm tra lại.</p>"
        self.nut_goi_y.disabled = not bool(self.goi_y)

    def ap_dung_goi_y(self, _=None):
        for cot, (value, _) in self.goi_y.items():
            self.truong[cot].value = value

    def thong_tin_them(self):
        return {**{cot: o.value for cot, o in self.truong.items() if cot not in ("gearbox", "region_name")},
                "co_mo_ta": int(self.co_mo_ta.value), **{cot: int(o.value) for cot, o in self.co.items()}}

    def du_doan(self):
        return self.dinh_gia(self.hang.value, self.dong.value, self.nam.value, self.odo.value,
                            self.truong["gearbox"].value, self.truong["region_name"].value,
                            gia_rao=self.gia.value.strip() or None, thong_tin_them=self.thong_tin_them())

    def hien_ket_qua(self, _=None):
        try:
            result = self.du_doan()
        except (ValueError, TypeError) as exc:
            self.ket_qua.value = '<p role="alert">Dữ liệu chưa hợp lệ: ' + escape(str(exc)) + "</p>"
            return
        thong_so = {"Hãng": self.hang.value, "Dòng": self.dong.value, "Năm": self.nam.value,
                    "ODO": self.odo.value, "Giá rao (triệu)": self.gia.value or "Không nhập",
                    **{o.description.rstrip(":"): o.value for o in self.truong.values()},
                    "Có mô tả": self.co_mo_ta.value,
                    **{TEN_CO[c]: o.value for c, o in self.co.items()}}
        mo_ta = tom_tat_mo_ta(self.thong_tin_them())
        nguon = "Có mô tả chi tiết" if mo_ta["co_mo_ta"] else "Chỉ có tiêu đề hoặc người dùng không nhập mô tả"
        self.ket_qua.value = "<h3>Kết quả</h3>" + "".join(
            f"<p><b>{escape(str(k))}:</b> {escape(str(v))}</p>" for k, v in result.items()
        ) + "<h3>Thông tin người bán công bố</h3>" + (
            f"<p><b>Nguồn văn bản:</b> {escape(nguon)}</p>"
            f"<p><b>Số cam kết tìm thấy:</b> {mo_ta['so_cam_ket']}</p>"
            f"<p><b>Nội dung:</b> {escape(mo_ta['noi_dung'])}</p>"
            "<p>Thông tin này do người bán công bố, chưa phải tình trạng xe đã được kiểm chứng.</p>"
        ) + "<details><summary>Thông số đã dùng</summary>" + "".join(
            f"<p>{escape(str(k))}: {escape(str(v))}</p>" for k, v in thong_so.items()
        ) + "</details><p>Giá rao tham khảo; cần kiểm tra xe thực tế. Khoảng ±MAE không phải khoảng tin cậy.</p>"
