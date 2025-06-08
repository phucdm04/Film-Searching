# Preprocessing

Họ và tên: Nguyễn Thuận Phát

MSSV: 22280062

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
```

- Đưa các biến môi trường vào file .env theo hướng dẫn. Có thể tự tùy chỉnh tên collection nếu muốn.

- Chạy hàm main bằng lệnh:
```
python main.py
```

- Sau đó truy cập vào MongoDB Compass (local) hoặc MongoDB Atlas (cloud) để xem kết quả. Lưu ý, phải truy cập đúng database tương ứng với uri ban đầu.
