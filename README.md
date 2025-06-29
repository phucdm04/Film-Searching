<!-- 

## 📥Data Retrieval Script

To keep our credentials secure, we use a `.env` file to store the MongoDB connection URI.
```env
MONGO_URI=[your_mongodb_uri]
```

Then use the below script to retrieve data.
```python
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the MongoDB URI from environment variable
uri = os.getenv("MONGO_URI")

client = MongoClient(
    uri,
    tls=True,
    tlsAllowInvalidCertificates=True
)

db = client["Film"]
collection = db["Data"]
cursor = collection.find({}, {"_id": 0})
df = pd.DataFrame(list(cursor))
``` -->

🎬LSA Movie Web
=

Đây là một ứng dụng web sử dụng Latent Semantic Analysis (LSA) để tìm kiếm các bộ phim dựa trên nội dung mô tả.

# 📖Hướng dẫn
## 1️⃣Cài đặt các thư viện cần thiết
Trước tiên, cài đặt các thư viện yêu cầu:

```bash
pip install -r requirements.txt
```

## 2️⃣Tạo file `.env`
```
# MongoDB config
MONGO_URI=
DATABASE_NAME=Film
COLLECTION_NAME=Data
LSA_COLLECTION_NAME=lsa_svd_preprocessed
WEMB_COLLECTION_NAME=word_embedding_preprocessed

# Qdrant config
QDRANT_URL=
QDRANT_KEY=
```

## 3️⃣Chạy ứng dụng
```python
python run.py
```
Sau khi chạy thành công, mở trình duyệt và truy cập: http://127.0.0.1:5000

# 📃Cấu trúc thư mục gốc
```
root/
├── run.py                         # File chính để chạy ứng dụng Flask
├── requirements.txt               # Thư viện cần thiết
├── .env                           # lưu trữ các biến môi trường (xem ví dụ trong .env.example)
├── templates/                     # Giao diện HTML
├── static/                        # Tài nguyên tĩnh (ảnh, CSS, JS)
├── database_connector/            # Kết nối đến cơ sở dữ liệu (MongoDB, v.v.)
├── embedding/                     # Thư mục chứa file .py của các mô hình
├── preprocessing/                 # Tiền xử lý dữ liệu
├── search_engine/                 # Logic xử lý truy vấn tìm kiếm
├── trained_models/                # Các model .pkl

```
**Lưu ý:** để xem mã code của các mô hình, hãy vào folder `embedding/`
