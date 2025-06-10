# search_engine/search_logic.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from dotenv import load_dotenv
from database_connector.qdrant_connector import connect_to_qdrant, search_points
from preprocessing.preprocessing import WordEmbeddingPipeline, LSASVDPipeline
from embedding.ppmi import PPMIEmbedder, TruncatedSVD 



load_dotenv()

# Kết nối Qdrant và nạp mô hình + pipeline
url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_KEY")
client = connect_to_qdrant(url, key)

import numpy as np

def embed_with_glove(tokens, embedder):
    vectors = []

    for token in tokens:
        if token in embedder['word2id']:
            idx = embedder['word2id'][token]
            vec = embedder['embeddings'][idx]
            vectors.append(vec)

    if not vectors:
        return np.zeros(embedder['embeddings'].shape[1])  # fallback

    return np.mean(vectors, axis=0)



def search_query(query_text, model_name):
    with open(f"./trained_models/{model_name}.pkl", 'rb') as f:
        print(model_name)
        embedder = pickle.load(f)
        
    # Chọn pipeline tương ứng
    if model_name == "tfidf":
        pipeline = LSASVDPipeline()
        processed_query = pipeline.preprocess(query_text)
        # Tiền xử lý + vector hóa
        embedded_query = embedder.transform_docs([processed_query])[0] # add [0] to make sure shape is (n, )
        results = search_points(client, model_name, embedded_query)

    elif model_name == "hellinger_pca":
        pipeline = WordEmbeddingPipeline()
        processed_query = pipeline.preprocess_single_text(query_text)
        # Tiền xử lý + vector hóa
        embedded_query = embedder.transform_docs([processed_query])[0] # add [0] to make sure shape is (n, )
        results = search_points(client, model_name, embedded_query)

    elif model_name == "bow":
        pipeline = LSASVDPipeline()
        processed_query = pipeline.preprocess(query_text)
        embedded_query = embedder.transform([processed_query])[0] # add [0] to make sure shape is (n, )
        collection_name = "svd_bow"
        results = search_points(client, collection_name, embedded_query)


        # raise ValueError(f"Model chưa được hỗ trợ.")
    elif model_name == "glove":
        pipeline = WordEmbeddingPipeline()
        processed_query = pipeline.preprocess_single_text(query_text)  # list of tokens
        embedded_query = embed_with_glove(processed_query, embedder)

        # raise ValueError(f"Model chưa được hỗ trợ.")
    elif model_name == "ppmi":
        pipeline = WordEmbeddingPipeline()
        processed_query = pipeline.preprocess_single_text(query_text)  # ví dụ: list token sau tiền xử lý
        embedded_query = embedder.transform_docs([processed_query])[0]  # giả sử có transform_docs
        collection_name = "ppmi"  # tên collection Qdrant bạn dùng
        results = search_points(client, collection_name, embedded_query)
        # raise ValueError(f"Model chưa được hỗ trợ.")
    elif model_name == "fasttext":
        raise ValueError(f"Model chưa được hỗ trợ.")
    elif model_name == "word2vec":
        raise ValueError(f"Model chưa được hỗ trợ.")
    else:
        raise ValueError(f"Model chưa được hỗ trợ.")

    # results = search_points(client, model_name, embedded_query)


    return results
