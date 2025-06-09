import os
import pickle
from dotenv import load_dotenv
from database_connector.qdrant_connector import connect_to_qdrant
from database_connector.mongodb_connector import connect_to_mongodb, get_all_documents, get_collection, get_database
from preprocessing.data_models import QdrantPoint
from qdrant_client.models import PointStruct
from embedding.ppmi import PPMIEmbedder, TruncatedSVD 

# Load biến môi trường
load_dotenv()

# 1. Kết nối Qdrant
qdrant_client = connect_to_qdrant(os.getenv("QDRANT_URL"), os.getenv("QDRANT_KEY"))
print("✅ Connected to Qdrant!")

# 2. Kết nối MongoDB
mongo_client = connect_to_mongodb(os.getenv("MONGO_URI"))
database = get_database(mongo_client, os.getenv("DATABASE_NAME"))
collection = get_collection(database, os.getenv("LSA_COLLECTION_NAME"))
all_documents = get_all_documents(collection)
print(f"✅ Total documents: {len(all_documents)}")

# 3. Chuẩn hoá văn bản
def normalize_text_field(doc):
    directors_raw = doc['metadata'].get('directors')
    writers_raw = doc['metadata'].get('writers')

    directors = " ".join([d.strip().lower() for d in directors_raw.split(',')]) if directors_raw else ""
    writers = " ".join([w.strip().lower() for w in writers_raw.split(',')]) if writers_raw else ""
    description = doc.get('cleaned_description', '').lower() if doc.get('cleaned_description') else ""

    return f"{directors} {writers} {description}"

corpus = []
raw_descriptions = []

for doc in all_documents:
    cleaned = doc.get('cleaned_description', '').lower()
    raw = doc.get('original_description', '')  

    corpus.append(cleaned)
    raw_descriptions.append(raw)


print(f"Corpus length: {len(corpus)}")
print(f'Test some samples: ')
# corpus[:5]

# 4. Load mô hình PPMI đã huấn luyện
model_path = "./trained_models/ppmi.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# 5. Lấy ma trận PPMI cho tất cả từ
embedding_matrix = model.transform(corpus)  # (vocab_size, embedding_dim)
print(f"✅ Generated embedding matrix of shape: {embedding_matrix.shape}")

# 6. Tạo embedding cho từng văn bản trong corpus
import numpy as np

def get_text_embedding(text, model, embedding_matrix):
    words = text.split()
    vectors = []
    for w in words:
        if w in model.vocab:
            idx = model.vocab[w]
            vec = embedding_matrix[idx]
            vectors.append(vec)
    if len(vectors) == 0:
        return np.zeros(embedding_matrix.shape[1])
    else:
        return np.mean(vectors, axis=0)

embedding_matrix_for_corpus = np.array([get_text_embedding(text, model, embedding_matrix) for text in corpus])
print(f"✅ Generated embedding matrix for corpus: {embedding_matrix_for_corpus.shape}")

# 7. Tạo danh sách QdrantPoint
list_of_points = []
for idx in range(len(corpus)):
    point = QdrantPoint(
        text=raw_descriptions[idx],  # ← chính là description gốc
        vector=embedding_matrix_for_corpus[idx],
        metadata=all_documents[idx]['metadata']
    )
    list_of_points.append(point)

print(f"✅ Prepared {len(list_of_points)} QdrantPoint objects.")


# 7. Đẩy dữ liệu vào Qdrant (tuỳ chọn)
from qdrant_client.http import models as rest


from qdrant_client.models import Distance, VectorParams

collection_name = "ppmi_svd_2"  # đổi tên tùy ý

# Tạo collection nếu chưa có
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=len(list_of_points[0].vector),
            distance=Distance.COSINE,
        ),
    )
    print(f"✅ Created collection '{collection_name}'")
from qdrant_client import QdrantClient
from httpx import Timeout

# timeout = Timeout(timeout=120.0)  # 120 giây timeout
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_KEY"),
    timeout=120.0
)



batch_size = 400
for i in range(0, len(list_of_points), batch_size):
    batch_points = list_of_points[i:i+batch_size]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=i + j,  # dùng i + j để id không bị trùng ở các batch
                vector=point.vector,
                payload={
                    "text": point.text,
                    "metadata": point.metadata
                }
            )
            for j, point in enumerate(batch_points)
        ]
    )


print("✅ Data upserted to Qdrant!")


