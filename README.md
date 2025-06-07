# Film-Searching


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
```

## 🎬 LSA Movie Web

### 1. Cài đặt các thư viện cần thiết

Trước tiên, hãy cài đặt các thư viện yêu cầu:

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
Mở Terminal 
```bash
python run.py
```
 Sau khi chạy thành công, bạn sẽ thấy dòng như sau:

 ```bash
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
Mở trình duyệt và truy cập: http://127.0.0.1:5000 hoặc truy cập link ngrok(để share với người khác)

### 3. Cấu trúc thư mục chính
```
Film-Searching/
├── run.py                    # File khởi chạy Flask app
├── templates/                # Giao diện HTML
├── static/                   # CSS, JS, hình ảnh
├── search_engine/            # Logic xử lý tìm kiếm
├── trained_models/           # Chứa mô hình .pkl
├── embedding/                # Các vector đã tính
├── requirements.txt          # Thư viện cần thiết
└── README.md                 # File hướng dẫn
```

search_engine/search_logic.py là nơi định nghĩa logic xử lý truy vấn tìm kiếm của web app

