# Preprocessing
## Cấu trúc folder

```
preprocessing
|--- main.py                    # File main dùng để run toàn bộ progress
|--- data_models.py             # File config và validate các trường dữ liệu của data. Nhằm đồng bộ, chuẩn hóa các trường dữ liệu
|--- preprocessing.py           # File config các Class cho việc preprocessing data
|--- README.md               
```
## Hướng dẫn
Để thực hiện việc tiền xử lý dữ liệu, chạy hàm main bằng lệnh:
```
python main.py
```
**Lưu ý:** Phải có file `.env` ở đường truyền gốc để sử dụng API.
