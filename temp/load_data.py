from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict, Any

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions
from typing import List, Optional, Set

class LSASVDPipeline:
    """
    Preprocessing pipeline for LSA/SVD
    """

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    html_pattern = re.compile(r'<[^>]+>')
    non_alpha_pattern = re.compile(r'[^a-zA-Z\s]')

    def __init__(
        self,
        extra_stopwords: Optional[Set[str]] = None
    ):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        base_sw = set(stopwords.words('english'))
        self.stop_words = base_sw.union(extra_stopwords or set())

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        # lowercase
        text = text.lower()
        # remove HTML
        text = self.html_pattern.sub('', text)
        # remove URLs
        text = self.url_pattern.sub('', text)
        # remove non-alphabetic
        text = self.non_alpha_pattern.sub(' ', text)
        # normalize spaces
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize_filter(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        return [tok for tok in tokens if tok not in self.stop_words and len(tok) > 2]

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(tok) for tok in tokens]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(tok) for tok in tokens]

    def preprocess(self, text: str, use_stemming: bool = True) -> str:
        cleaned = self.clean_text(text)
        tokens = self.tokenize_filter(cleaned)
        processed = self.stem(tokens) if use_stemming else self.lemmatize(tokens)
        return ' '.join(processed)

    def batch(self, texts: List[str], use_stemming: bool = True) -> List[str]:
        return [self.preprocess(txt, use_stemming) for txt in texts]

class WordEmbeddingPipeline:
    """
    Preprocessing pipeline for WordEmbedding (Word2Vec/GloVe/FastText)
    --> CORE: Lightweight preprocessing. Maintain context.
    """

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    html_pattern = re.compile(r'<[^>]+>')
    punct_except_basic = re.compile(r'[^\w\s\.\!?]')

    def __init__(self, minimal_stopwords: Optional[Set[str]] = None):
        self.lemmatizer = WordNetLemmatizer()
        defaults = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were"}
        self.stop_words = defaults.union(minimal_stopwords or set())

    def expand_contractions(self, text: str) -> str:
        return contractions.fix(text)

    def clean_gentle(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self.expand_contractions(text)
        text = text.lower()
        text = self.html_pattern.sub('', text)
        text = self.url_pattern.sub(' ', text)
        text = self.punct_except_basic.sub(' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize_sentences(self, text: str) -> List[List[str]]:
        sentences = sent_tokenize(text)
        result = []
        for sent in sentences:
            toks = word_tokenize(sent)
            filtered = [tok for tok in toks if tok not in self.stop_words and len(tok) > 1]
            if filtered:
                result.append(filtered)
        return result

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(tok) for tok in tokens]

    def preprocess(self, text: str) -> List[List[str]]:
        cleaned = self.clean_gentle(text)
        sents = self.tokenize_sentences(cleaned)
        return [self.lemmatize(sent) for sent in sents]

    def preprocess_single_text(self, text: str) -> List[str]:
        sents = self.preprocess(text)
        return [tok for sent in sents for tok in sent]

    def flatten(self, text: str) -> str:
        return ' '.join(self.preprocess_single_text(text))

    def batch(self, texts: List[str]) -> List[List[List[str]]]:
        return [self.preprocess(txt) for txt in texts]

class MongoDBClient:
    def __init__(self, mongo_uri: str, database_name: str):
        """
        Initialize the MongoDBClient and connect to the database.
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.client = self._connect()
        self.database = self._get_database()

    def _connect(self) -> MongoClient:
        """
        Connect to MongoDB server and ping the deployment.
        """
        client = MongoClient(self.mongo_uri)
        try:
            client.admin.command("ping")
            print("Pinged your deployment. Successfully connected to MongoDB!")
        except Exception as e:
            raise Exception(f"Connection failed: {e}")
        return client

    def _get_database(self) -> Database:
        """
        Get the specified database from the MongoDB client.
        """
        try:
            db = self.client[self.database_name]
            print(f"Database '{self.database_name}' connected successfully!")
        except Exception as e:
            raise Exception(f"Getting database failed: {e}")
        return db

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection from the MongoDB database.
        """
        try:
            collection = self.database[collection_name]
            print(f"Collection '{collection_name}' connected successfully!")
        except Exception as e:
            raise Exception(f"Getting collection failed: {e}")
        return collection

    def get_all_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        Get all documents from a specified MongoDB collection.
        """
        try:
            collection = self.get_collection(collection_name)
            documents = list(collection.find())
            return documents
        except Exception as e:
            raise Exception(f"Fetching documents failed: {e}")



# def connect_to_mongodb(mongo_uri: str) -> MongoClient:
#     """
#     Connect to MongoDB server.
#     """
#     mongo_client = MongoClient(mongo_uri)
#     try:
#         mongo_client.admin.command("ping")
#         print("Pinged your deployment. Successfully connected to MongoDB!")
#     except Exception as e:
#         raise Exception(f"Connection failed: {e}")
#     return mongo_client

# def get_database(mongo_client: MongoClient, database_name: str) -> Database:
#     """
#     Get a database from the MongoDB client.
#     """
#     try:
#         database = mongo_client[database_name]
#         print(f"Database '{database_name}' connected successfully!")
#     except Exception as e:
#         raise Exception(f"Getting database failed: {e}")
#     return database

# def get_collection(database: Database, collection_name: str) -> Collection:
#     """
#     Get a collection from the MongoDB database.
#     """
#     try:
#         collection = database[collection_name]
#         print(f"Collection '{collection_name}' connected successfully!")
#     except Exception as e:
#         raise Exception(f"Getting collection failed: {e}")
#     return collection

# def get_all_documents(collection: Collection) -> List[Dict[str, Any]]:
#     """
#     Get all documents from a MongoDB collection.
#     """
#     all_documents = []
#     try:
#         documents = collection.find()
#         for document in documents:
#             all_documents.append(document)
#     except Exception as e:
#         raise Exception(f"Fetching documents failed: {e}")
#     return all_documents


# def load_data(COLLECTION_NAME: Optional[str] = None) -> List[Dict[str, Any]]:
#     """
#     Load all documents from a MongoDB collection.
#     """
#     load_dotenv()

#     MONGO_URI = os.getenv("MONGO_URI")
#     if MONGO_URI is None:
#         raise ValueError("MONGO_URI not found in environment variables.")
    
#     DATABASE_NAME = "Film"
#     if COLLECTION_NAME is None:
#         COLLECTION_NAME = "Data"

#     mongo_client = connect_to_mongodb(MONGO_URI)
#     database = get_database(mongo_client, DATABASE_NAME)
#     collection = get_collection(database, COLLECTION_NAME)
#     all_documents = get_all_documents(collection)

#     return all_documents


if __name__ == "__main__":
    pass
