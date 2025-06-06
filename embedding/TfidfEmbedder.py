import math
import numpy as np
from collections import Counter
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
# from sklearn.decomposition import TruncatedSVD

class TruncatedSVD:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None  # ma trận thành phần chính (eigenvectors)
        self.singular_values_ = None  # singular values
        self.explained_variance_ratio_ = None
        self.fitted = False
        

    def fit(self, X: np.ndarray):
        # X dạng ma trận (samples x features)
        # Trung bình hoá X theo chiều features
        self.mean_ = np.mean(X, axis=0)

        # SVD đầy đủ
        U, S, VT = np.linalg.svd(X, full_matrices=False)

        # Giữ lại n_components đầu
        if self.n_components is None:
            self.n_components = X.shape[1]
        self.components_ = VT[:self.n_components, :]
        self.singular_values_ = S[:self.n_components]

        # Tính explained variance ratio
        total_var = np.sum(S ** 2)
        comp_var = S[:self.n_components] ** 2
        self.explained_variance_ratio_ = comp_var / total_var

        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model chưa fit dữ liệu!")
        
        # Chiếu dữ liệu lên các thành phần chính
        return np.dot(X, self.components_.T)

    def choose_n_components(self, threshold=0.95):
        """
        Chọn số thành phần chính sao cho tỉ lệ phương sai tích lũy >= threshold (mặc định 95%)
        """
        if not self.fitted:
            raise ValueError("PCA must be fitted first!")
        cum_var_ratio = np.cumsum(self.explained_variance_ratio_)

        # Tìm chỉ số thành phần thỏa điều kiện
        n_components = np.searchsorted(cum_var_ratio, threshold) + 1
        self.n_components = n_components
        self.components_ = self.components_[:n_components]  # Cắt lại components

        return n_components

    def plot_cumulative_variance(self, threshold = 0.95):
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("Bạn cần gọi .fit() trước khi vẽ biểu đồ explained variance.")

        cum_var = np.cumsum(self.explained_variance_ratio_)
        plt.figure(figsize=(20, 10))
        
        plt.plot(range(1, len(cum_var)+1), cum_var, linestyle='-')
        if threshold is not None:
            n_component = self.choose_n_components(threshold)
            plt.axvline(x=n_component, color='blue', linestyle='--', label=f'Selected Components = {n_component}')
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Remained Information = {threshold * 100}%')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance Ratio")
        plt.title("Cumulative Explained Variance by PCA Components")
        plt.grid(True)
        plt.legend()
        plt.show()

class TEmbedder:
    def __init__(
        self,
        smooth_idf: bool = True,
        norm: Optional[str] = 'l2',
        n_components: Optional[int] = None,
        max_features: Optional[int] = None
    ):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.smooth_idf = smooth_idf
        self.norm = norm  # 'l1', 'l2', or None
        self.max_features = max_features
        self.lsa = TruncatedSVD(n_components)
        self.doc_embeddings: List[np.ndarray] = []
        self.raw_documents: List[str] = []

    def _create_tfidf_matrix(self, documents: List[str]) -> np.ndarray:
        """Tính TF-IDF và chuẩn hóa"""
        tfidf_matrix = np.zeros((len(documents), len(self.vocab)))

        for i, doc in enumerate(tqdm(documents, desc="Creating Tf-Idf matrix...")):
            tokens = doc.lower().split()
            tf = Counter(tokens)
            doc_len = len(tokens)

            for word in tf:
                if word in self.vocab:
                    tf_val = tf[word] / doc_len
                    idf_val = self.idf[word]
                    tfidf_matrix[i, self.vocab[word]] = tf_val * idf_val

        # Chuẩn hóa
        if self.norm == 'l2':
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            tfidf_matrix = tfidf_matrix / norms
        elif self.norm == 'l1':
            norms = np.sum(np.abs(tfidf_matrix), axis=1, keepdims=True)
            norms[norms == 0] = 1
            tfidf_matrix = tfidf_matrix / norms
        elif self.norm is None:
            pass
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

        return tfidf_matrix

    
    def fit(self, documents: List[str], plot: bool = False) -> None:
        """Xây dựng từ vựng và tính IDF"""
        N = len(documents)
        df = Counter()

        for doc in documents:
            tokens = set(doc.lower().split())
            for token in tokens:
                df[token] += 1

        # Giới hạn từ vựng theo max_features
        if self.max_features is not None:
            most_common = df.most_common(self.max_features)
            vocab_words = [word for word, _ in most_common]
        else:
            vocab_words = list(df.keys())

        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}

        if self.smooth_idf:
            self.idf = {
                word: math.log((1 + N) / (1 + df[word])) + 1
                for word in self.vocab
            }
        else:
            self.idf = {
                word: math.log(N / df[word])
                for word in self.vocab
            }

        tfidf_matrix = self._create_tfidf_matrix(documents)
        self.lsa.fit(tfidf_matrix)
        if plot:
            self.lsa.plot_cumulative_variance()

    def transform_docs(self, documents: List[str]) -> np.ndarray:
        tfidf_matrix = self._create_tfidf_matrix(documents)
        return self.lsa.transform(tfidf_matrix)


if __name__ == "__main__":
    docs = [
        "cat eats fish",
        "dog chases cat",
        "fish swims in river",
        "cat and dog play together"
    ]

    query = "cat chases mouse"
    # Chỉ lấy 5 từ phổ biến nhất và giảm chiều còn 2
    embedder = TEmbedder(max_features=None, n_components=None)
    embedder.fit(docs, plot=True)
    X = embedder.transform(docs)

    print("Shape:", X.shape)  # -> (4, 2)
    print("Vocab:", embedder.vocab)
    print(X)
    print(embedder.transform([query]))