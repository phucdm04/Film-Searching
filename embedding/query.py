# Xử lý môi trường
import os
import sys
import pickle
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()


import pickle
from database_connector.qdrant_connector import connect_to_qdrant, get_collection, ensure_collection_exists, create_collection, insert_point_to_qdrant, delete_point, search_points
from tqdm import tqdm
from database_connector.mongodb_connector import load_documents
from qdrant_client.http.models import PointStruct
from preprocessing.Preprocessor import LSASVDPipeline, WordEmbeddingPipeline


def upsert(qdrant_client, collection_name, embedded_docs, mongo_collection):

    for i, docs in enumerate(tqdm(embedded_docs, desc=f"Upserting to Collection {collection_name}")):
        metadata = mongo_collection[i]["metadata"]
        metadata["mongo_id"] =  mongo_collection[i]["id"]
        point = [PointStruct(id=i, vector=docs, payload={
            "text":  mongo_collection[i]["original_description"],
            "metadata": metadata,
            })]
        qdrant_client.upsert(
            collection_name=collection_name,
            points = point
        )

if __name__ == "__main__":
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_KEY")
    client = connect_to_qdrant(url, key)

    query = "Sally Temple actress from theatre helps people of land, pretends to be a lady to deceive lord, runs away, fights with enemy, kidnapped by Duke of Chatto, rescued by Romsey, gives Pump Lane to her people"
    # id should be returned tt0008779
    # load model
    model_name = "hellinger_pca" # or "tfidf"
    with open(f"./embedding/trained_models/{model_name}.pkl", 'rb') as f:
        embedder = pickle.load(f)
    
    # load preprocess pipeline
    pipeline = WordEmbeddingPipeline() # change to LSASVDPipline if query with TFIDF
    processed_query = pipeline.preprocess_single_text(query)
    embedded_query = embedder.transform_docs([processed_query])[0] # add [0] to make sure shape is (n, )

    # search
    print(search_points(client, model_name, embedded_query))
    





    
    