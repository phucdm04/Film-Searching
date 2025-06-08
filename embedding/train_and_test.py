# Xử lý môi trường
import os
import sys
import pickle
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

# import các pack
from database_connector.mongodb_connector import load_documents
from embedding.HellingerPcaEmbedder import HPEmbedder
from embedding.TfidfEmbedder import TEmbedder
from preprocessing.Preprocessor import WordEmbeddingPipeline, LSASVDPipeline

def train_helling_pca(docs, max_features: int = None, save_path: str = None, find_suit_n_component: bool = True):
    if max_features is None:
        max_features = int(np.sqrt(len(docs)))

    embedder = HPEmbedder(n_components = None, max_features=max_features)
    embedder.fit(docs)
    if find_suit_n_component:
        best_n = embedder.find_best_n_components(plot=True)
        print(f"Re-embedding HellingerPCA with {best_n} components:")
        embedder = HPEmbedder(n_components=best_n, max_features=max_features)
        embedder.fit(docs)
        
    path = "./embedding/trained_models/tfidf.pkl" if save_path is None else save_path
    with open(save_path, "wb") as f:
        pickle.dump(embedder, f)
        print(f"Dump sucessfully embedder into {save_path}!")

    return embedder

def train_tfidf(docs, max_features: int = None, save_path: str = None, find_suit_n_component: bool = True):
    if max_features is None:
        max_features = int(np.sqrt(len(docs)))
        
    embedder = TEmbedder(n_components = None, max_features=max_features)
    embedder.fit(docs)
    if find_suit_n_component:
        best_n = embedder.find_best_n_components(plot=False)
        print(f"Re-embedding TFIDF with {best_n} components:")
        TEmbedder()
        embedder = TEmbedder(n_components=best_n, max_features=max_features)
        embedder.fit(docs)

    save_path = "./embedding/trained_models/tfidf.pkl" if save_path is None else save_path
    with open(save_path, "wb") as f:
        pickle.dump(embedder, f)
        print(f"Dump sucessfully embedder into {save_path}!")

    return embedder

if __name__ == "__main__":
    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("DATABASE_NAME")
    
    lsa_collection_name = os.getenv("LSA_COLLECTION_NAME")
    lsa_docs = load_documents(uri, db_name, lsa_collection_name)
    processed_lsa_docs = [doc["cleaned_description"] for doc in lsa_docs]

    embedder = train_tfidf(processed_lsa_docs, max_features=1830)   # 1423 

    # wemb_collection_name = os.getenv("WEMB_COLLECTION_NAME")
    # wemb_docs = load_documents(uri, db_name, wemb_collection_name)
    # processed_wemb_docs = [doc["cleaned_description"] for doc in wemb_docs]

    # embedder = train_helling_pca(processed_wemb_docs, max_features=1266) # 656
    
    path = "./embedding/trained_models/tfidf.pkl"
    with open(path, "wb") as f:
        pickle.dump(embedder, f)
        print("Dump sucessfully!")