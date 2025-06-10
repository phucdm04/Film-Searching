# --- Phase 1: Train Word2Vec and Save Embeddings ---
import numpy as np
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def get_sentences():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    data_list = list(collection.find({}, {"cleaned_description": 1}))
    return [doc['cleaned_description'] for doc in data_list if 'cleaned_description' in doc]


def train_word2vec(sentences, embedding_dim=100, window_size=2, learning_rate=0.01, epochs=1):
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

    with open('word2vec_embedding.pkl', 'wb') as f:
        pickle.dump({'embedding': W, 'word2idx': word2idx, 'idx2word': idx2word}, f)


# --- Phase 2: Use Embedding for Similarity Matching ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def get_vector(tokens, embedding_matrix, word2idx):
    vectors = [embedding_matrix[word2idx[t]] for t in tokens if t in word2idx]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_matrix.shape[1])


def find_similar_films(new_description, top_k=5):
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    with open('word2vec_embedding.pkl', 'rb') as f:
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

    for _id, score in top_matches:
        film = collection.find_one({'id': _id})
        print(f"\n🎬 Film: {film['metadata']['film_name']}")
        print(f"📝 Description: {film['original_description']}")
        print(f"📊 Similarity: {score:.4f}")


# --- Run full pipeline ---
if __name__ == '__main__':
    # Training
    sentences = get_sentences()
    train_word2vec(sentences)

    # Similarity Matching
    query = "A mother fakes her death and returns in disguise to see her children."
    find_similar_films(query)
