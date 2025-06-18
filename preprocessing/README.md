<<<<<<< HEAD
<<<<<<<< HEAD:preprocessing/README.md
# Preprocessing
=======
# Preprocessing

Họ và tên: Nguyễn Thuận Phát

MSSV: 22280062

>>>>>>> MPhuc-branch
## Cấu trúc folder

```
preprocessing
|
|--- main.py                    # File main dùng để run toàn bộ progress
|--- mongodb_connector.py       # File config các method liên quan đến mongodb: connect, get documents, get data,...
|--- data_models.py             # File config và validate các trường dữ liệu của data. Nhằm đồng bộ, chuẩn hóa các trường dữ liệu
|--- preprocessing.py           # File config các Class cho việc preprocessing data
|--- notebook_experiments.py    # File chạy thử nghiệm và test các chức năng
|--- .env                       # File dùng để chứa các biến môi trường
|--- requirements.txt           # File chứa các thư viện cần thiết để chạy code   
|--- readme.md               
```
## Hướng dẫn

- Chạy lệnh dưới đây để tải các thư viện cần thiết.
```
pip install -r requirements.txt
<<<<<<< HEAD
========
# Film-Searching


## 📥 Data Retrieval Script

To keep our credentials secure, we use a `.env` file to store the MongoDB connection URI.
```env
MONGO_URI=[your_mongodb_uri]
>>>>>>>> MPhuc-branch:README.md
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
=======
```

- Đưa các biến môi trường vào file .env theo hướng dẫn. Có thể tự tùy chỉnh tên collection nếu muốn.

- Chạy hàm main bằng lệnh:
```
python main.py
```

- Sau đó truy cập vào MongoDB Compass (local) hoặc MongoDB Atlas (cloud) để xem kết quả. Lưu ý, phải truy cập đúng database tương ứng với uri ban đầu.
>>>>>>> MPhuc-branch
