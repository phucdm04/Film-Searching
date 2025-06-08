import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Union, Optional
from tqdm import tqdm
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

from database_connector.mongodb_connector import load_documents


class TruncatedSVD:
    def __init__(self, n_components: Optional[int] = None) -> None:
        self.n_components: Optional[int] = n_components
        self.components_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X: np.ndarray) -> None:
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

        if self.n_components is None:
            self.n_components = X.shape[1]

        self.components_ = VT[:self.n_components]
        self.singular_values_ = S[:self.n_components]
        total_var = np.sum(S ** 2)
        comp_var = S[:self.n_components] ** 2
        self.explained_variance_ratio_ = comp_var / total_var
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TruncatedSVD not fit.")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def choose_n_components(self, threshold: float = 0.95) -> int:
        if not self.fitted:
            raise RuntimeError("Model does not fit data!")
        cum_var = np.cumsum(self.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum_var, threshold) + 1)
        self.n_components = n_comp
        self.components_ = self.components_[:n_comp]
        self.singular_values_ = self.singular_values_[:n_comp]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_comp]
        return n_comp

    def plot_cumulative_variance(self, threshold: float = 0.95) -> None:
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("Call .fit() before plotting.")
        cum_var = np.cumsum(self.explained_variance_ratio_)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='-')
        if threshold is not None:
            n_components = np.searchsorted(cum_var, threshold) + 1
            plt.axvline(x=n_components, color='blue', linestyle='--', label=f'{n_components} components')
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'{int(threshold*100)}% variance')
        plt.title("Cumulative Explained Variance by LSA Components (PPMI + LSA)")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


class PPMIEmbedder:
    def __init__(self, window_size: int = 2, max_features: Optional[int] = None, n_components: Optional[int] = None) -> None:
        self.window_size: int = window_size
        self.max_features: Optional[int] = max_features
        self.n_components: Optional[int] = n_components
        self.vocab: Dict[str, int] = {}
        self.svd: TruncatedSVD = TruncatedSVD(n_components)
        self.embeddings: Optional[np.ndarray] = None
        self._original_ppmi: Optional[np.ndarray] = None

    def _tokenize(self, doc: Union[str, List[str]]) -> List[str]:
        return doc.split() if isinstance(doc, str) else doc

    def _build_vocab(self, docs: List[Union[str, List[str]]]) -> None:
        counter = Counter()
        for doc in docs:
            counter.update(self._tokenize(doc))
        if self.max_features:
            most_common = counter.most_common(self.max_features)
            self.vocab = {word: i for i, (word, _) in enumerate(most_common)}
        else:
            self.vocab = {word: i for i, word in enumerate(counter.keys())}

    def _build_cooc_matrix(self, docs: List[Union[str, List[str]]]) -> np.ndarray:
        size = len(self.vocab)
        matrix = np.zeros((size, size), dtype=np.float64)
        for doc in tqdm(docs, desc="Build a Co-occurrence Matrix"):
            tokens = self._tokenize(doc)
            token_ids = [self.vocab[t] for t in tokens if t in self.vocab]
            for i, center in enumerate(token_ids):
                start = max(i - self.window_size, 0)
                end = min(i + self.window_size + 1, len(token_ids))
                for j in range(start, end):
                    if i == j:
                        continue
                    context = token_ids[j]
                    matrix[center, context] += 1.0
        return matrix

    def _calculate_ppmi(self, M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        total = np.sum(M)
        row_sums = np.sum(M, axis=1, keepdims=True)
        col_sums = np.sum(M, axis=0, keepdims=True)
        p_wc = M / total
        p_w = row_sums / total
        p_c = col_sums / total
        with np.errstate(divide='ignore'):
            pmi = np.log((p_wc + eps) / (p_w @ p_c + eps))
        return np.maximum(pmi, 0)

    def fit(self, docs: List[Union[str, List[str]]]) -> None:
        self._build_vocab(docs)
        cooc = self._build_cooc_matrix(docs)
        ppmi = self._calculate_ppmi(cooc)
        self.embeddings = self.svd.fit_transform(ppmi)
        self._original_ppmi = ppmi

    def transform(self, docs: List[Union[str, List[str]]]) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("You need to call fit() first.")
        cooc = self._build_cooc_matrix(docs)
        ppmi = self._calculate_ppmi(cooc)
        return self.svd.transform(ppmi)

    def transform_docs(self, docs: List[Union[str, List[str]]]) -> np.ndarray:
        if self.embeddings is None or not self.vocab:
            raise RuntimeError("You need to call fit() first.")
        dim = self.embeddings.shape[1]
        doc_vectors: List[np.ndarray] = []
        for doc in docs:
            tokens = self._tokenize(doc)
            vectors = [self.embeddings[self.vocab[t]] for t in tokens if t in self.vocab]
            doc_vec = np.mean(vectors, axis=0) if vectors else np.zeros(dim)
            doc_vectors.append(doc_vec)
        return np.array(doc_vectors)

    def choose_n_components(self, threshold: float = 0.95) -> int:
        n_comp = self.svd.choose_n_components(threshold)
        if self._original_ppmi is not None:
            self.embeddings = self.svd.transform(self._original_ppmi)
        return n_comp


def train_ppmi_lsa(
    docs: List[str],
    max_features: Optional[int] = None,
    save_path: Optional[str] = None,
    find_suit_n_component: bool = True
) -> PPMIEmbedder:
    print(f"Starting PPMI training on {len(docs)} documents...")
    if max_features is None:
        max_features = int(np.sqrt(len(docs)))
        print(f"Max_feature not specified, use sqrt(N): {max_features}")
    else:
        print(f"Using max_features = {max_features}")
    embedder = PPMIEmbedder(window_size=4, n_components=None, max_features=max_features)
    start_time = time.time()
    embedder.fit(docs)
    print(f"- Vocabulary size: {len(embedder.vocab)}")
    print(f"- PPMI matrix shape:: {embedder._original_ppmi.shape}")
    print(f"- Initial PPMI computation completed in {time.time() - start_time:.2f} seconds")
    if find_suit_n_component:
        print("Selecting suitable number of components based on 95% variance threshold...")
        best_n = embedder.choose_n_components(threshold=0.95)
        print(f"- Suitable number of components for 95% variance: {best_n}")
        print("Re-training with optimal number of components...")
        embedder = PPMIEmbedder(n_components=best_n, max_features=max_features)
        start_time = time.time()
        embedder.fit(docs)
        print(f"Re-training completed in {time.time() - start_time:.2f} seconds")
    save_path = "./embedding/trained_models/ppmi.pkl" if save_path is None else save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(embedder, f)
        print(f"Embedder saved to: {save_path}")
    return embedder


if __name__ == "__main__":
    print("Connecting to MongoDB...")
    uri: str = os.getenv("MONGO_URI")
    db_name: str = os.getenv("DATABASE_NAME")
    lsa_collection_name: str = os.getenv("LSA_COLLECTION_NAME")
    lsa_docs = load_documents(uri, db_name, lsa_collection_name)
    processed_lsa_docs: List[str] = [doc["cleaned_description"] for doc in lsa_docs]
    print(f"Loaded {len(processed_lsa_docs)} documents from MongoDB.")
    embedder = train_ppmi_lsa(
        processed_lsa_docs,
        max_features=1200,
        save_path="./embedding/trained_models/ppmi.pkl",
        find_suit_n_component=True
    )
    print("PPMI training pipeline completed.")
