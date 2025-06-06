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


if __name__ == "__main__":
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_KEY")
    client = connect_to_qdrant(url, key)

    query = "actress at Drury Lane helps people on Pump Lane against a duke, then pretends to be someone else's wife to save her neighbors – a lord finds out and tries to force her to marry him, but a prize fighter helps her escape. Story about a woman rescuing a neighborhood and a duke trying to oppress them, with a fake marriage and a fight."
    # id should be returned 
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
    





    
    