# search_engine/search_logic.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from dotenv import load_dotenv
from database_connector.qdrant_connector import connect_to_qdrant, search_points
from preprocessing.preprocessing import WordEmbeddingPipeline, LSASVDPipeline



load_dotenv()

# Kết nối Qdrant và nạp mô hình + pipeline
url = os.getenv("QDRANT_URL")
key = os.getenv("QDRANT_KEY")
client = connect_to_qdrant(url, key)

# model_name = "hellinger_pca"
# model_name = "tfidf"



# pipeline = LSASVDPipeline()
# pipeline = WordEmbeddingPipeline()

def search_query(query_text, model_name):
    with open(f"./trained_models/{model_name}.pkl", 'rb') as f:
        print(model_name)
        embedder = pickle.load(f)

    # processed_query = pipeline.preprocess(query_text)
    # embedded_query = embedder.transform_docs([processed_query])[0]
    # results = search_points(client, model_name, embedded_query)

    # Chọn pipeline tương ứng
    if model_name == "tfidf":
        pipeline = LSASVDPipeline()
        processed_query = pipeline.preprocess(query_text)
    elif model_name == "hellinger_pca":
        pipeline = WordEmbeddingPipeline()
        processed_query = pipeline.preprocess_single_text(query_text)
    else:
        raise ValueError(f"Model chưa được hỗ trợ.")

    # Tiền xử lý + vector hóa
    embedded_query = embedder.transform_docs([processed_query])[0] # add [0] to make sure shape is (n, )
    results = search_points(client, model_name, embedded_query)

    return results

# def search_query(query_text):
#     processed_query = pipeline.preprocess(query_text)
#     embedded_query = embedder.transform_docs([processed_query])[0]
#     results = search_points(client, model_name, embedded_query)
#     return results
# def search_query(query_text):
#     processed_query = preprocessor.preprocess_single_text(query_text)
#     joined_query = ' '.join(processed_query)
#     embedded_query = embedder.transform_docs([joined_query])[0]
#     results = search_points(client, model_name, embedded_query)
#     return results
