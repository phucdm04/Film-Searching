import numpy as np
from collections import Counter
from typing import List, Dict, Union
from tqdm import tqdm
import matplotlib.pyplot as plt

class TruncatedSVD:
    def __init__(self, n_components: int = None):
        self.n_components = n_components
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.fitted = False

    def fit(self, X: np.ndarray):
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

    def transform(self, X: np.ndarray):
        if not self.fitted:
            raise RuntimeError("TruncatedSVD not fit.")
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    def choose_n_components(self, threshold=0.95):
        if not self.fitted:
            raise RuntimeError("Model does not fit data!")
        cum_var = np.cumsum(self.explained_variance_ratio_)
        n_comp = np.searchsorted(cum_var, threshold) + 1
        self.n_components = n_comp
        self.components_ = self.components_[:n_comp]
        self.singular_values_ = self.singular_values_[:n_comp]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_comp]
        return n_comp

    def plot_cumulative_variance(self, threshold=0.95):
        """
        Vẽ biểu đồ thể hiện tỷ lệ phương sai tích lũy của các thành phần chính.
        Nếu threshold được đặt, sẽ vẽ đường ngang và dọc minh họa số chiều cần giữ lại.
        """
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("You need to call .fit() before plotting the explained variance.")

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
    def __init__(self, window_size: int = 2, max_features: int = None, n_components: int = None):
        self.window_size = window_size
        self.max_features = max_features
        self.n_components = n_components
        self.vocab: Dict[str, int] = {}
        self.svd = TruncatedSVD(n_components)
        self.embeddings = None

    def _tokenize(self, doc: Union[str, List[str]]):
        return doc.split() if isinstance(doc, str) else doc

    def _build_vocab(self, docs: List[Union[str, List[str]]]):
        counter = Counter()
        for doc in docs:
            counter.update(self._tokenize(doc))
        if self.max_features:
            most_common = counter.most_common(self.max_features)
            self.vocab = {word: i for i, (word, _) in enumerate(most_common)}
        else:
            self.vocab = {word: i for i, word in enumerate(counter.keys())}

    def _build_cooc_matrix(self, docs: List[Union[str, List[str]]]):
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

    def _calculate_ppmi(self, M: np.ndarray, eps: float = 1e-8):
        total = np.sum(M)
        row_sums = np.sum(M, axis=1, keepdims=True)
        col_sums = np.sum(M, axis=0, keepdims=True)

        p_wc = M / total
        p_w = row_sums / total
        p_c = col_sums / total

        with np.errstate(divide='ignore'):
            pmi = np.log((p_wc + eps) / (p_w @ p_c + eps))
        ppmi = np.maximum(pmi, 0)
        return ppmi

    def fit(self, docs: List[Union[str, List[str]]]):
        self._build_vocab(docs)
        cooc = self._build_cooc_matrix(docs)
        ppmi = self._calculate_ppmi(cooc)
        self.embeddings = self.svd.fit_transform(ppmi)
        self._original_ppmi = ppmi

    def transform(self, docs: List[Union[str, List[str]]]):
        if self.embeddings is None:
            raise RuntimeError("You need to call fit() first.")
        cooc = self._build_cooc_matrix(docs)
        ppmi = self._calculate_ppmi(cooc)
        return self.svd.transform(ppmi)

    def transform_docs(self, docs: List[Union[str, List[str]]]):
        if self.embeddings is None or not self.vocab:
            raise RuntimeError("You need to call fit() first.")
        dim = self.embeddings.shape[1]
        doc_vectors = []
        for doc in docs:
            tokens = self._tokenize(doc)
            vectors = [self.embeddings[self.vocab[t]] for t in tokens if t in self.vocab]
            if vectors:
                doc_vec = np.mean(vectors, axis=0)
            else:
                doc_vec = np.zeros(dim)
            doc_vectors.append(doc_vec)
        return np.array(doc_vectors)

    def choose_n_components(self, threshold=0.95):
        n_comp = self.svd.choose_n_components(threshold)
        if self._original_ppmi is not None:
            self.embeddings = self.svd.transform(self._original_ppmi)
        return n_comp


if __name__ == "__main__":
    docs = [
        "A is father of B",
        "C is mother of B",
        "A loves C",
        "B is a child of A and C",
        "B is son of A"
    ]

    embedder = PPMIEmbedder(window_size=2, n_components=6)
    embedder.fit(docs)
    embedder.svd.plot_cumulative_variance(threshold=0.95)

    print("\n Word Embeddings:")
    for word, idx in embedder.vocab.items():
        print(f"{word:10}: {embedder.embeddings[idx]}")

    doc_embeddings = embedder.transform_docs(docs)
    print("\n Document Embeddings Shape:", doc_embeddings.shape)
    print(doc_embeddings)
