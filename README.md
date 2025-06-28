# Film-Searching
<!-- 

## 📥 Data Retrieval Script

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

# 🎬 LSA Movie Web

Đây là một ứng dụng web sử dụng Latent Semantic Analysis (LSA) để tìm kiếm các bộ phim dựa trên nội dung mô tả.

## 🚀 Cách chạy ứng dụng

### 1. Cài đặt các thư viện cần thiết

Trước tiên, cài đặt các thư viện yêu cầu:

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
```python
python run.py
```
Sau khi chạy thành công, bạn sẽ thấy dòng như sau:
```
Successfully connected to Qdrant!
 * Ngrok public URL: NgrokTunnel: "https://***.ngrok-free.app" -> "http://localhost:5000"
 * Serving Flask app 'run'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```
Mở trình duyệt và truy cập: http://127.0.0.1:5000 hoặc https://***.ngrok-free.app (có thể chia sẻ cho người khác)

### 3. Cấu trúc thư mục chính
```
lsa_movie_web/
├── run.py                          # File chính để chạy ứng dụng Flask
├── requirements.txt               # Thư viện cần thiết
├── templates/                     # Giao diện HTML
├── static/                        # Tài nguyên tĩnh (ảnh, CSS, JS)
│   └── Image/
├── database_connector/            # Kết nối đến cơ sở dữ liệu (MongoDB, v.v.)
├── embedding/                     # Thư mục xử lý embedding & mô hình
│   └── bow_svd_model/             # Mô hình biểu diễn BoW + SVD
├── preprocessing/                 # Xử lý dữ liệu trước khi phân tích
├── search_engine/                 # Logic xử lý truy vấn tìm kiếm
├── .env/                          # lưu trữ các biến môi trường (xem ví dụ trong .env.example)
├── trained_models/                # Các model .pkl
```
