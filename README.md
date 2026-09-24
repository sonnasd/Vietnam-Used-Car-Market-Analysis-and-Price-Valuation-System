# Vietnam Used Car Market Analysis and Price Valuation System

Du an mon hoc Data Analytics Project (DAP391m) - He thong dinh gia va phan tich thi truong xe o to cu tai Viet Nam.

Tai lieu chuan cua du an la [docs/proposal DAP.docx](docs/proposal%20DAP.docx) (da duoc GVHD review). Tien do va phan cong theo tung task: [docs/Tracking_DAP391m.xlsx](docs/Tracking_DAP391m.xlsx).

## 1. Muc tieu va san pham ban giao

Muc tieu:

- Tham dinh gia doc lap: mo hinh hoi quy uoc luong gia thi truong hop ly tu thong so ky thuat, tuoi xe va do hao mon.
- Phat hien Deal: doi chieu gia niem yet voi gia du doan de phan loai tin dang thanh Deal hoi / Re bat thuong, Gia hop ly, Het gia cao.

San pham ban giao:

- Bo du lieu da chuan hoa.
- Notebook phan tich gom ban do nhiet 63 tinh, Word Cloud va dashboard tuong tac.
- Mo hinh dinh gia dong goi `.pkl`.
- Logic phat hien Deal.

Bao cao duoc viet truc tiep trong cac Jupyter notebook: moi notebook gom code Python va phan markdown giai thich cho tung buoc.

## 2. Phan cong trach nhiem

- Do Khanh Duy (Data & ML Engineer):
  - Phu trach thu muc: 01_crawler_and_ml/ (nhanh feature/crawler-and-ml)
  - Nhiem vu: Thu thap du lieu tu API Cho Tot, cao bo sung mo ta dai theo list_id (notebook 01); xay dung, danh gia va dong goi mo hinh dinh gia (notebook 05).
- Quach Thien Nhan (Data Analyst & Dashboard Engineer):
  - Phu trach thu muc: 02_analytics_and_dashboard/ (nhanh feature/data-cleaning-dashboard)
  - Nhiem vu: Lam sach, chuan hoa du lieu va loc ngoai lai IQR (notebook 02); EDA, ban do nhiet 63 tinh va dashboard tuong tac trong notebook (notebook 03).
- Nguyen Huy Son (NLP, Deal Detection & Project Coordinator):
  - Phu trach thu muc: 03_nlp_and_deal/ va docs/ (nhanh feature/nlp-and-deal)
  - Nhiem vu: Phan tich van ban, co nhi phan va Word Cloud (notebook 04); phat hien Deal va kich ban dinh gia dau-cuoi (notebook 06); notebook tong quan (notebook 00), quan ly tien do va slide bao ve.

## 3. Cau truc thu muc

```text
Vietnam-Used-Car-Market-Analysis-and-Price-Valuation-System/
|-- 01_crawler_and_ml/                 Do Khanh Duy
|   |-- crawler/
|   |   `-- scraper.py                 crawler API Cho Tot
|   |-- 01_thu_thap_du_lieu.ipynb
|   `-- 05_mo_hinh_dinh_gia.ipynb
|-- 02_analytics_and_dashboard/        Quach Thien Nhan
|   |-- 02_lam_sach_du_lieu.ipynb
|   `-- 03_phan_tich_truc_quan.ipynb
|-- 03_nlp_and_deal/                   Nguyen Huy Son
|   |-- 00_tong_quan.ipynb
|   |-- 04_phan_tich_van_ban.ipynb
|   `-- 06_phat_hien_deal.ipynb
|-- data/
|   |-- raw/
|   |   `-- cars.csv                   du lieu goc: 15.000 tin x 35 truong
|   |-- processed/                     du lieu da xu ly, dung chung
|   `-- geojson/                       ban do 63 tinh/thanh
|-- models/                            mo hinh .pkl
|-- docs/
|   |-- proposal DAP.docx
|   `-- Tracking_DAP391m.xlsx
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## 4. Thu tu doc notebook

| # | Notebook | Noi dung | Phu trach |
|---|---|---|---|
| 00 | 03_nlp_and_deal/00_tong_quan.ipynb | Van de, muc tieu, du lieu, ket luan | Nguyen Huy Son |
| 01 | 01_crawler_and_ml/01_thu_thap_du_lieu.ipynb | Thu thap du lieu tu Cho Tot | Do Khanh Duy |
| 02 | 02_analytics_and_dashboard/02_lam_sach_du_lieu.ipynb | Lam sach, chuan hoa, loc ngoai lai | Quach Thien Nhan |
| 03 | 02_analytics_and_dashboard/03_phan_tich_truc_quan.ipynb | EDA, ban do nhiet, dashboard | Quach Thien Nhan |
| 04 | 03_nlp_and_deal/04_phan_tich_van_ban.ipynb | Co nhi phan, Word Cloud | Nguyen Huy Son |
| 05 | 01_crawler_and_ml/05_mo_hinh_dinh_gia.ipynb | Huan luyen, danh gia mo hinh | Do Khanh Duy |
| 06 | 03_nlp_and_deal/06_phat_hien_deal.ipynb | Phat hien Deal, kich ban dinh gia dau-cuoi | Nguyen Huy Son |

Luong du lieu giua cac notebook:

- data/raw/cars.csv -> 02 -> data/processed/cars_cleaned.csv -> 03, 04, 05, 06
- 01 -> data/raw/descriptions.csv -> 04 -> data/processed/text_flags.csv -> 05
- 05 -> models/car_pricing_model.pkl -> 06

## 5. Quy uoc lam viec

- Moi nguoi chi sua notebook trong thu muc cua minh: hai nguoi cung sua mot file .ipynb tren hai nhanh rat de xung dot khi merge.
- File dung chung (du lieu da xu ly, mo hinh) dat trong data/ va models/, khong dat trong thu muc ca nhan.
- Doc/ghi file qua bien ROOT o cell dau moi notebook, khong dung duong dan tuyet doi tren may ca nhan.
- Luu notebook kem output (bieu do, bang) de nguoi doc xem duoc ma khong can chay lai.
- Trong data/processed/ chi commit cars_cleaned.csv va text_flags.csv; cac file tam khac da bi .gitignore chan.
- File .pkl phai la pipeline hoan chinh (ma hoa + mo hinh) va nho hon 100 MB (gioi han cua GitHub).

## 6. Huong dan cai dat moi truong

Yeu cau Python 3.10 tro len.

Khoi tao moi truong ao:

```bash
python -m venv .venv
```

Kich hoat moi truong ao:

- Tren Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

- Tren Linux / macOS:
```bash
source .venv/bin/activate
```

Cai dat cac goi thu vien can thiet:

```bash
pip install -r requirements.txt
```

## 7. Quy chuan lam viec voi Git va phan nhanh

He thong phan nhanh gom 1 nhanh chinh va cac nhanh tinh nang doc lap:

- main: Nhanh chinh luu tru ma nguon cua toan bo du an.
- feature/crawler-and-ml: Nhanh lam viec cua Do Khanh Duy.
- feature/data-cleaning-dashboard: Nhanh lam viec cua Quach Thien Nhan.
- feature/nlp-and-deal: Nhanh lam viec cua Nguyen Huy Son.

Cac buoc lam viec chuan:

1. Chuyen sang nhanh ca nhan truoc khi code:
```bash
git checkout <ten-nhanh-ca-nhan>
```

2. Dong bo du lieu moi nhat tu nhanh main:
```bash
git pull origin main
```

3. Commit va push ma nguon len nhanh ca nhan:
```bash
git add .
git commit -m "Mo ta ngan gon noi dung thay doi"
git push origin <ten-nhanh-ca-nhan>
```

4. Tao Pull Request (PR) tren GitHub tu nhanh ca nhan vao nhanh main de cac thanh vien khac review truoc khi gop code.
