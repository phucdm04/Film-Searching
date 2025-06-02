import math
from collections import Counter
from typing import List, Optional
import numpy as np

class TfidfEmbedder:
    def __init__(self, smooth_idf: bool = True, norm: Optional[str] = 'l2'):
        self.vocab = {}
        self.idf = {}
        self.smooth_idf = smooth_idf
        self.norm = norm  # 'l1', 'l2', or None

    def fit(self, documents: List[str]) -> None:
        """Xây dựng từ vựng và tính IDF"""
        N = len(documents)
        df = Counter()

        for doc in documents:
            tokens = set(doc.lower().split())
            for token in tokens:
                df[token] += 1

        self.vocab = {word: idx for idx, word in enumerate(df.keys())}

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

    def transform(self, documents: List[str]) -> np.ndarray:
        """Tính TF-IDF và chuẩn hóa"""
        tfidf_matrix = np.zeros((len(documents), len(self.vocab)))

        for i, doc in enumerate(documents):
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

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)
