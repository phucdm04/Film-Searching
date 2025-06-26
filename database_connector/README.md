Database connector
=
# Cấu trúc folder

```
preprocessing
|--- __init.py__
|--- mongodb_connector.py       # File chứa các phương thức để kết nối tới MongoDB để lấy dữ liệu
|--- qdrant_connector.py		# File chứa các phương thức để kết nối và điều chỉnh với Qdrant
|--- README.md             
```
# Hướng dẫn
Cần chuẩn bị các API và các biến môi trường cần thiết chứa trong `.env` ở đường dẫn gốc. File `.env` có dạng như sau:
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
