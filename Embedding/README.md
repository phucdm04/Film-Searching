# About folder
Cấu trúc folder
```
embedding
|-- __init__.py
|-- HellingerPCAEmbedder.py
|-- TfidfEmbedder.py
|-- trained_models
	|-- tfidf.pkl 
	|-- hellinger_pca.pkl
|-- vectordb_configs.json 				# File configs khi truy cập vô qdrant để upsert dữ liệu
|-- README.md
```

Code gọi model
```python
import pickle

model_name = ""
with open(f"./trained_models/{model_name}", "rb") as f:
	embedder = pickle.load(f)
qdrant_client  =  QdrantClient(
	url="https://9c9b2577-14d6-49a3-b752-460ea063effe.eu-west-1-0.aws.cloud.qdrant.io:6333",
	api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.e7K6I6PWiaRH0D551h-pnO31dYJFG1rgyGpLMkj_Qd8",
)
```

Truy xuất từ Qdrant
```python
from qdrant_client import QdrantClient
qdrant_client = QdrantClient(
	url=,
	api_key=,)
# liên hệ Phúc để lấy url và api_key 🫢

def query_result(query: str, embedder, collection_name: str) -> None:
	query_embedding = embedder.transform([query])[0] # make sure query has been processed
	search_result = qdrant_client.search(
		collection_name=collection_name,
		query_vector=query_embedding,
		limit=10
	)

	for hit in search_result:
		print(f"Score: {hit.score:.3f} - Text:\n{hit.payload['text']}\n")
```

