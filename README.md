# Vietnam Used Car Market Analysis and Price Valuation System

Du an mon hoc Data Analytics Project (DAP391m) - He thong dinh gia va phan tich thi truong xe o to cu tai Viet Nam.

## 1. Phan cong trach nhiem

- Do Khanh Duy (Data & ML Engineer):
  - Phu trach thu muc: 01_crawler_and_ml/
  - Nhiem vu: Thu thap du lieu, tach dac trung NLP, huan luyen va danh gia mo hinh hoc may (Random Forest, XGBoost).
- Quach Thien Nhan (Data Analyst & Dashboard Engineer):
  - Phu trach thu muc: 02_analytics_and_dashboard/
  - Nhiem vu: Lam sach du lieu, xu ly ngoai lai IQR, truc quan hoa EDA, xay dung ban do nhiet Choropleth va ung dung Dashboard.
- Nguyen Huy Son (Conversational AI & Project Coordinator):
  - Phu trach thu muc: 03_chatbot_and_api/ va docs/
  - Nhiem vu: Thiet lap kich ban IBM Watson Assistant, xay dung Webhook FastAPI, quan ly tien do va tong hop cac ban bao cao Report 1 - 5.

## 2. Cau truc thu muc

```text
Vietnam-Used-Car-Market-Analysis-and-Price-Valuation-System/
|-- data/
|   |-- raw/
|   |   `-- cars.csv
|   |-- processed/
|   |   `-- .gitkeep
|   `-- geojson/
|       `-- .gitkeep
|-- docs/
|   |-- Ke_hoach_Du_an_DAP391m.docx
|   |-- Tracking_DAP391m.xlsx
|   `-- Tracking_Task.docx
|-- 01_crawler_and_ml/
|   |-- crawler/
|   |-- nlp_extraction/
|   |-- ml_pipeline/
|   `-- notebooks/
|-- 02_analytics_and_dashboard/
|   |-- data_cleaning/
|   |-- choropleth_map/
|   |-- dashboard/
|   `-- notebooks/
|-- 03_chatbot_and_api/
|   |-- watson_assistant/
|   |-- api/
|   `-- reports/
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## 3. Huong dan cai dat moi truong

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

## 4. Quy chuan lam viec voi Git va phan nhanh

He thong phan nhanh gom 2 nhanh goc va cac nhanh tinh nang doc lap:

- main: Nhanh luu tru phien ban on dinh nhat phuc vu bao cao cuoi ky.
- develop: Nhanh tich hop chung cho toan doi.
- feature/crawler-and-ml: Nhanh lam viec cua Do Khanh Duy.
- feature/data-cleaning-dashboard: Nhanh lam viec cua Quach Thien Nhan.
- feature/chatbot-and-fastapi: Nhanh lam viec cua Nguyen Huy Son.

Cac buoc lam viec chuan:

1. Chuyen sang nhanh ca nhan truoc khi code:
```bash
git checkout <ten-nhanh-ca-nhan>
```

2. Dong bo du lieu moi nhat tu nhanh develop:
```bash
git pull origin develop
```

3. Commit va push ma nguon len nhanh ca nhan:
```bash
git add .
git commit -m "Mo ta ngan gon noi dung thay doi"
git push origin <ten-nhanh-ca-nhan>
```

4. Tao Pull Request (PR) tren GitHub tu nhanh ca nhan vao nhanh develop de cac thanh vien khac review truoc khi gop code.
