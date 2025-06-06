import math
import numpy as np
from collections import Counter
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

class TruncatedSVD:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None
        self.fitted = False

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        n_dim = VT.shape[0]
        if self.n_components is None or self.n_componets > n_dim:
            self.n_components = n_dim

        self.components_ = VT[:self.n_components, :]
        self.singular_values_ = S[:self.n_components]

        total_var = np.sum(S ** 2)
        comp_var = S[:self.n_components] ** 2
        self.explained_variance_ratio_ = comp_var / total_var
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Fitted first!")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def choose_n_components(self, threshold=0.95):
        # Find number of components to keep based on threshold
        if not self.fitted:
            raise ValueError("Fitted first!")
        cum_var_ratio = np.cumsum(self.explained_variance_ratio_)
        n_components = np.searchsorted(cum_var_ratio, threshold) + 1
        self.n_components = n_components
        self.components_ = self.components_[:n_components]
        return n_components

    def plot_cumulative_variance(self, threshold = 0.95):
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("Call fit() !!!")
        cum_var = np.cumsum(self.explained_variance_ratio_)
        
        plt.figure(figsize=(20, 10))
        plt.plot(range(1, len(cum_var)+1), cum_var, linestyle='-')

        if threshold is not None:
            n_component = self.choose_n_components(threshold)
            plt.axvline(x=n_component, color='blue', linestyle='--', label=f'Selected Components = {n_component}')
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Remained Information = {threshold * 100}%')

        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("Cumulative Explained Variance by SVD Components")
        plt.grid(True)
        plt.legend()
        plt.show()


class FastText:
    def __init__(self, vector_size=50, window_size=2, epochs=5, lr=0.01, min_count=1):
        self.vector_size = vector_size
        self.window_size = window_size
        self.epochs = epochs
        self.lr = lr
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = []
        self.input_vectors = None
        self.output_vectors = None

    def build_vocab(self, corpus):
        # Build vocabulary from corpus
        word_freq = Counter(w for sentence in corpus for w in sentence)
        self.vocab = [w for w, c in word_freq.items() if c >= self.min_count]
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def generate_training_pairs(self, corpus):
        # Generate (center, context) pairs
        pairs = []
        for sentence in corpus:
            for i, center in enumerate(sentence):
                if center not in self.word2idx:
                    continue
                for j in range(max(0, i - self.window_size), min(len(sentence), i + self.window_size + 1)):
                    if i != j and sentence[j] in self.word2idx:
                        pairs.append((self.word2idx[center], self.word2idx[sentence[j]]))
        return pairs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, corpus):
        # Train embeddings using skip-gram
        self.build_vocab(corpus)
        vocab_size = len(self.vocab)
        self.input_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, self.vector_size))
        self.output_vectors = np.zeros((vocab_size, self.vector_size))
        training_pairs = self.generate_training_pairs(corpus)
        for epoch in range(self.epochs):
            np.random.shuffle(training_pairs)
            for center, context in training_pairs:
                v_in = self.input_vectors[center]
                v_out = self.output_vectors[context]
                score = self.sigmoid(np.dot(v_in, v_out))
                grad = self.lr * (1 - score)
                self.input_vectors[center] += grad * v_out
                self.output_vectors[context] += grad * v_in
            print(f"Epoch {epoch+1} finished")

    def get_vector(self, word):
        # Return vector for a word
        if word in self.word2idx:
            return self.input_vectors[self.word2idx[word]]
        return np.zeros(self.vector_size)

# Combine FastText + SVD into LSA pipeline
class FastTextLSAEmbedder:
    def __init__(
        self,
        n_components: Optional[int] = 2,
        vector_size: int = 50,
        window: int = 2,
        epochs: int = 5
    ):
        self.model = FastText(vector_size=vector_size, window_size=window, epochs=epochs)
        self.lsa = TruncatedSVD(n_components)

    def _preprocess(self, doc: str) -> List[str]:
        return doc.lower().split()

    def _embed_doc(self, tokens: List[str]) -> np.ndarray:
        # Average word vectors to get document vector
        vectors = [self.model.get_vector(w) for w in tokens if w in self.model.word2idx]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

    def fit(self, documents: List[str], plot: bool = False):
        # Train FastText and apply SVD
        tokenized = [self._preprocess(doc) for doc in documents]
        self.model.train(tokenized)
        X = np.array([self._embed_doc(doc) for doc in tokenized])
        self.lsa.fit(X)
        if plot:
            self.lsa.plot_cumulative_variance()

    def transform(self, documents: List[str]) -> np.ndarray:
        tokenized = [self._preprocess(doc) for doc in documents]
        X = np.array([self._embed_doc(doc) for doc in tokenized])
        return self.lsa.transform(X)

# Example usage
if __name__ == "__main__":
    docs = [
        "cat eats fish",
        "dog chases cat",
        "fish swims in river",
        "cat and dog play together"
    ]
    query = "cat chases mouse"

    embedder = FastTextLSAEmbedder(n_components=2)
    embedder.fit(docs, plot=True)
    print("Doc vectors:", embedder.transform(docs))
    print("Query vector:", embedder.transform([query]))
