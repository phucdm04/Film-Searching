import numpy as np
from collections import Counter
from typing import List, Dict, Optional

class BagOfWords:
    def __init__(self, 
                 min_word_freq: int = 1,
                 max_features: Optional[int] = None,
                 tokenizer: str = 'whitespace'):
        """
        Initialize Bag of Words model for cleaned string data
        
        Args:
            min_word_freq: Minimum frequency of a word to be kept
            max_features: Maximum number of features
            tokenizer: Tokenization method ('whitespace' or custom function)
        """
        self.min_word_freq = min_word_freq
        self.max_features = max_features
        self.tokenizer = tokenizer
        
        # Vocabulary and mappings
        self.vocabulary = {}  # word -> index
        self.idx_to_word = {}  # index -> word
        self.word_counts = Counter()
        self.is_fitted = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of words
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of words
        """
        if self.tokenizer == 'whitespace':
            # Split by whitespace and remove empty tokens
            return [token.strip() for token in text.split() if token.strip()]
        elif callable(self.tokenizer):
            # Use custom tokenizer function
            return self.tokenizer(text)
        else:
            raise ValueError("tokenizer must be 'whitespace' or a function")
    
    def fit(self, documents: List[str]) -> 'BagOfWords':
        """
        Learn vocabulary from a list of documents
        
        Args:
            documents: List of cleaned strings
        """
        # Reset
        self.word_counts = Counter()
        
        # Tokenize and count word frequencies
        for doc in documents:
            tokens = self._tokenize(doc)
            self.word_counts.update(tokens)
        
        # Filter words by min_word_freq
        filtered_words = {word: count for word, count in self.word_counts.items() 
                         if count >= self.min_word_freq}
        
        # Sort by descending frequency
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # Limit number of features
        if self.max_features:
            sorted_words = sorted_words[:self.max_features]
        
        # Create vocabulary
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        self.idx_to_word = {idx: word for word, idx in self.vocabulary.items()}
        
        self.is_fitted = True
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Convert documents to BoW matrix
        
        Args:
            documents: List of cleaned strings
            
        Returns:
            BoW matrix with shape (n_documents, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fit. Call fit() first.")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        
        # Initialize BoW matrix
        bow_matrix = np.zeros((n_docs, n_features), dtype=int)
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            word_counts = Counter(tokens)
            
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    bow_matrix[doc_idx, word_idx] = count
        
        return bow_matrix
    
    def transform_single(self, document: str) -> np.ndarray:
        """
        Convert a single document to BoW vector
        
        Args:
            document: A cleaned string
            
        Returns:
            BoW vector with shape (n_features,)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fit.")
        
        n_features = len(self.vocabulary)
        bow_vector = np.zeros(n_features, dtype=int)
        
        tokens = self._tokenize(document)
        word_counts = Counter(tokens)
        
        for word, count in word_counts.items():
            if word in self.vocabulary:
                word_idx = self.vocabulary[word]
                bow_vector[word_idx] = count
        
        return bow_vector
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self) -> List[str]:
        """Return the list of words in the vocabulary"""
        if not self.is_fitted:
            raise ValueError("Model has not been fit.")
        return [self.idx_to_word[i] for i in range(len(self.vocabulary))]
    
    def get_vocabulary_size(self) -> int:
        """Return the size of the vocabulary"""
        return len(self.vocabulary)
    
    def get_word_frequency(self, word: str) -> int:
        """Return the frequency of a word in the corpus"""
        return self.word_counts.get(word, 0)
    
    def get_word_index(self, word: str) -> Optional[int]:
        """Return the index of a word in the vocabulary"""
        return self.vocabulary.get(word, None)
    
    def get_document_tokens(self, document: str) -> List[str]:
        """Return tokens of a document (for debugging)"""
        return self._tokenize(document)
    
    def print_vocabulary_info(self, top_n: int = 10):
        """Print information about the vocabulary"""
        if not self.is_fitted:
            print("Model has not been fit.")
            return
        
        print(f"Vocabulary size: {self.get_vocabulary_size()}")
        print(f"Total words in corpus: {sum(self.word_counts.values())}")
        print(f"Unique words in corpus: {len(self.word_counts)}")
        print(f"Top {top_n} most common words:")
        
        for i, word in enumerate(self.get_feature_names()[:top_n]):
            freq = self.get_word_frequency(word)
            print(f"  {i+1:2d}. '{word:15s}': {freq:4d}")
    
    def analyze_document(self, document: str):
        """Detailed analysis of a document"""
        if not self.is_fitted:
            raise ValueError("Model has not been fit.")
        
        tokens = self._tokenize(document)
        word_counts = Counter(tokens)
        bow_vector = self.transform_single(document)
        
        print(f"Document: '{document[:100]}...' (length: {len(document)})")
        print(f"Tokens: {tokens[:20]}... (total: {len(tokens)})")
        print(f"Unique words: {len(word_counts)}")
        print(f"Words in vocabulary: {sum(1 for w in word_counts if w in self.vocabulary)}")
        print(f"BoW vector sum: {bow_vector.sum()}")
        
        # Top words in document
        in_vocab_words = [(w, c) for w, c in word_counts.most_common(10) 
                         if w in self.vocabulary]
        if in_vocab_words:
            print("Top words in vocabulary:")
            for word, count in in_vocab_words:
                idx = self.vocabulary[word]
                print(f"  '{word}': {count} (index: {idx})")
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file"""
        if not self.is_fitted:
            raise ValueError("Model has not been fit.")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for word, idx in self.vocabulary.items():
                freq = self.get_word_frequency(word)
                f.write(f"{word}\t{idx}\t{freq}\n")
        print(f"Vocabulary has been saved to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from file"""
        self.vocabulary = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    word, idx, freq = parts[0], int(parts[1]), int(parts[2])
                    self.vocabulary[word] = idx
                    self.idx_to_word[idx] = word
                    self.word_counts[word] = freq
        
        self.is_fitted = True
        print(f"Vocabulary has been loaded from {filepath}")