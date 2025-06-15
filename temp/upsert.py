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

def process_and_upsert(docs, model, qdrant_client, qdrant_collection_name, overwrited = True):
    processed_docs = [doc["cleaned_description"] for doc in docs]
    vectors = model.transform_doc(processed_docs)
    vector_size = len(vectors[0])

    
    if not ensure_collection_exists(qdrant_client, qdrant_collection_name, vector_size=vector_size):
        create_collection(qdrant_client, qdrant_collection_name, vector_size=vector_size)
    else:
        if overwrited:
            print(f"Recreate collection {qdrant_collection_name}")
            qdrant_client.delete_collection(collection_name=qdrant_collection_name)
            create_collection(qdrant_client, qdrant_collection_name, vector_size=vector_size)

    for i, _ in enumerate(tqdm(processed_docs, desc="Upserting")):
        
        qdrant_point = {
            "text": docs[i]['original_description'],
            "metadata": docs[i]['metadata'],
            "id":i,
            "vector":vectors[i],
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
    processed_docs = [doc["cleaned_description"] for doc in docs]

    with open("./trained_models/hellinger_pca.pkl", "rb") as f:
        embedder = pickle.load(f)

    process_and_upsert(docs[:], embedder, client, "hellinger_pca")
    # vectors = embedder.transform_doc(processed_docs)





    
    