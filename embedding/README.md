Documents
=
# Cấu trúc folder
```
embedding
|-- __init__.py
|-- README.md
|-- vectordb_configs.json
|-- HellingerPCAEmbedder.py
|-- TfidfEmbedder.py
|-- FastText.py
|-- word2Vec.py
|-- glove.py
|-- bow_svd_model
	|-- ...
|-- trained_models
	|-- tfidf.pkl 
	|-- hellinger_pca.pkl
```

## `HellingerPcaEmbedder.py`
Các methods chính:
- `fit`: cho Embedder học ma trận từ đồng xuất hiện
- `transform_word`: nhúng một từ
- `transform_docs`: nhúng tài liệu
- `find_best_n_components`: tìm giá trị thành phần chính giữ lại $i$% thông tin so với tài liệu gốc (mặc định là 95%)

## `TfidfEmbedder.py`
Các methods chính:
- `fit`: cho Embedder học ma trận từ đồng xuất hiện
- `transform_docs`: nhúng tài liệu
- `find_best_n_components`: tìm giá trị thành phần chính giữ lại $i$% thông tin so với tài liệu gốc (mặc định là 95%)


