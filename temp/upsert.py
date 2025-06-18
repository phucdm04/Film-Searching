# Xử lý môi trường
import os
import sys
import pickle
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

import pickle
from database_connector.qdrant_connector import connect_to_qdrant, get_collection, ensure_collection_exists, create_collection, insert_point_to_qdrant
from tqdm import tqdm
from database_connector.mongodb_connector import load_documents
from preprocessing.preprocessing import LSASVDPipeline, WordEmbeddingPipeline
from embedding.glove import GloVe
from qdrant_client.http.exceptions import UnexpectedResponse


def recreate_collection(client, collection_name, vector_size):
    # Xóa nếu collection đã tồn tại
    try:
        if ensure_collection_exists(client, collection_name, vector_size):
            print(f"Collection '{collection_name}' exists. Deleting it...")
            client.delete_collection(collection_name=collection_name)
    except UnexpectedResponse as e:
        print(f"Error when deleting collection: {e}")

    # Tạo mới collection
    print(f"Creating collection '{collection_name}'...")
    create_collection(client, collection_name, vector_size)


def process_and_upsert(docs, model, qdrant_client, qdrant_collection_name):
    vectors = model.encode(docs)
    vector_size = vectors.shape[0]
    
    recreate_collection(qdrant_client, qdrant_collection_name, vector_size=vector_size)


    for i, _ in enumerate(tqdm(docs, desc="Upserting")):
        vector = model.encode(docs[i]['cleaned_description'])
        if i == 0:
            print(f"Vector size: {vector.shape}, Vector norm: {np.linalg.norm(vector)}")
            print(f"Vector: {vector}...")  # Print first 5 elements for brevity
        
        qdrant_point = {
            "text": docs[i]['original_description'],
            "metadata": docs[i]['metadata'],
            "id":i,
            "vector": vector,
        }


        insert_point_to_qdrant(qdrant_client, qdrant_collection_name, qdrant_point)
        


if __name__ == "__main__":
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_KEY")
    client = connect_to_qdrant(url, key)

    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("DATABASE_NAME")
    
    collection_name = os.getenv("WEMB_COLLECTION_NAME")
    docs = load_documents(uri, db_name, collection_name)
    
    # processed_docs = [doc["cleaned_description"] for doc in docs]

    # with open("./trained_models/glove.pkl", "rb") as f:
    #     embedder = pickle.load(f)

    embedder = GloVe.from_pretrained("./trained_models/glove.pkl")

    process_and_upsert(docs, embedder, client, 'glove')
    # # vectors = embedder.transform_doc(processed_docs)