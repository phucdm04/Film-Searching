# word2vec_pipeline.py
import numpy as np
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import random
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# ---  Load sentences from MongoDB ---
def get_sentences():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    data_list = list(collection.find({}, {"cleaned_description": 1}))
    return [doc['cleaned_description'] for doc in data_list if 'cleaned_description' in doc]

# --- Train Word2Vec and save model ---
def train_word2vec(sentences, embedding_dim=500, window_size=2, learning_rate=0.01, epochs=1, save_path='word2vec_embedding.pkl'):
    vocab = set(word for sent in sentences for word in sent)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(word2idx)
    W = np.random.uniform(-0.01, 0.01, (vocab_size, embedding_dim))

    for _ in range(epochs):
        for sent in sentences:
            for idx, target in enumerate(sent):
                if target not in word2idx:
                    continue
                target_idx = word2idx[target]
                start = max(0, idx - window_size)
                end = min(len(sent), idx + window_size + 1)
                for context_pos in range(start, end):
                    if context_pos == idx:
                        continue
                    context_word = sent[context_pos]
                    if context_word not in word2idx:
                        continue
                    context_idx = word2idx[context_word]
                    error = W[target_idx] - W[context_idx]
                    W[target_idx] -= learning_rate * error
                    W[context_idx] += learning_rate * error

    with open(save_path, 'wb') as f:
        pickle.dump({'embedding': W, 'word2idx': word2idx, 'idx2word': idx2word}, f)

# --- Vector helpers ---
def get_vector(tokens, embedding_matrix, word2idx):
    vectors = [embedding_matrix[word2idx[t]] for t in tokens if t in word2idx]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_matrix.shape[1])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# ---  Similarity matching ---
def find_similar_films(new_description, top_k=5, model_path='word2vec_embedding.pkl'):
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    embedding_matrix = model_data['embedding']
    word2idx = model_data['word2idx']

    tokens = new_description.lower().split()
    new_vector = get_vector(tokens, embedding_matrix, word2idx)

    similarities = []
    for doc in collection.find({}, {"id": 1, "cleaned_description": 1}):
        vec = get_vector(doc['cleaned_description'], embedding_matrix, word2idx)
        sim = cosine_similarity(new_vector, vec)
        similarities.append((doc['id'], sim))

    top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for _id, score in top_matches:
        film = collection.find_one({'id': _id})
        print(f"\n Film: {film['metadata']['film_name']}")
        print(f" Description: {film['original_description']}")
        print(f" Similarity: {score:.4f}")

# ---  Evaluation with Silhouette Score ---
def choose_k(n_samples):
    return max(2, int(np.sqrt(n_samples)))

if __name__ == '__main__':
    #  Load cleaned descriptions
    sentences = get_sentences()
    train_word2vec(sentences, embedding_dim=500, window_size=2, learning_rate=0.01, epochs=5, save_path='word2vec.pkl')
    #  Test similarity search
    print("\n Testing Similarity Search")
    query = "Thomas Brainerd, Sr., as a prospector, is a dutiful and loving husband and father. Two children, Gertrude and Thomas, Jr., are born while the Brainerds live in a log cabin in the mountains"
    find_similar_films(query, model_path='word2vec.pkl')  
