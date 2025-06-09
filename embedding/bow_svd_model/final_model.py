from typing import List, Optional, Dict, Union
import numpy as np
from embedding.bow_svd_model.bow import BagOfWords
from embedding.bow_svd_model.dim_reduc import SVDModel

class BOW_SVD_Embedding:
    """
    Embedding pipeline = Bag-of-Words + SVD

    Args:
        - bow_args (Optional[Dict] = None): Config for BagOfWords
            + min_word_freq: min frequency of a word, defalut = 1
            + max_features: max features for bow, default = None
            + tokenizer: strategy to split sentence into single words, default = 'whitespace'
        - dim_reduc_args (Optional[Dict] = None): Config for SVDModel
            + n_components: number of features after reduction, default = 100
    """
    def __init__(
        self,
        bow_args: Optional[Dict] = None,
        dim_reduc_args: Optional[Dict] = None,
    ):
        self.bow_args = bow_args or {}
        self.dim_reduc_args = dim_reduc_args or {}

        self.bow = self._init_bow()
        self.dim_reduc = self._init_dim_reduc()
        self._is_fitted = False

    def _init_bow(self) -> BagOfWords:
        return BagOfWords(
            min_word_freq=self.bow_args.get("min_word_freq", 1),
            max_features=self.bow_args.get("max_features", None),
            tokenizer=self.bow_args.get("tokenizer", "whitespace")
        )

    def _init_dim_reduc(self) -> SVDModel:
        return SVDModel(
            n_components=self.dim_reduc_args.get("n_components", 100)
        )

    def fit(self, documents: Union[List[str], str]) -> "BOW_SVD_Embedding":
        texts = [documents] if isinstance(documents, str) else documents
        bow_matrix = self.bow.fit_transform(texts)
        self.dim_reduc.fit(bow_matrix)
        self._is_fitted = True
        return self

    def transform(self, documents: Union[List[str], str]) -> Union[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fit before calling transform.")

        is_single = isinstance(documents, str)
        texts = [documents] if is_single else documents

        bow_matrix = self.bow.transform(texts)
        reduced = self.dim_reduc.transform(bow_matrix)

        return reduced[0] if is_single else reduced

    def fit_transform(self, documents: Union[List[str], str]) -> Union[np.ndarray, np.ndarray]:
        texts = [documents] if isinstance(documents, str) else documents
        bow_matrix = self.bow.fit_transform(texts)
        reduced = self.dim_reduc.fit_transform(bow_matrix)
        self._is_fitted = True
        return reduced[0] if isinstance(documents, str) else reduced

    def get_feature_names(self) -> List[str]:
        return self.bow.get_feature_names()

    def get_vocabulary_size(self) -> int:
        return self.bow.get_vocabulary_size()
