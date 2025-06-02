from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv
import os
from typing import Optional, List, Dict, Any


def connect_to_mongodb(mongo_uri: str) -> MongoClient:
    """
    Connect to MongoDB server.
    """
    mongo_client = MongoClient(mongo_uri)
    try:
        mongo_client.admin.command("ping")
        print("Pinged your deployment. Successfully connected to MongoDB!")
    except Exception as e:
        raise Exception(f"Connection failed: {e}")
    return mongo_client


def get_database(mongo_client: MongoClient, database_name: str) -> Database:
    """
    Get a database from the MongoDB client.
    """
    try:
        database = mongo_client[database_name]
        print(f"Database '{database_name}' connected successfully!")
    except Exception as e:
        raise Exception(f"Getting database failed: {e}")
    return database


def get_collection(database: Database, collection_name: str) -> Collection:
    """
    Get a collection from the MongoDB database.
    """
    try:
        collection = database[collection_name]
        print(f"Collection '{collection_name}' connected successfully!")
    except Exception as e:
        raise Exception(f"Getting collection failed: {e}")
    return collection


def get_all_documents(collection: Collection) -> List[Dict[str, Any]]:
    """
    Get all documents from a MongoDB collection.
    """
    all_documents = []
    try:
        documents = collection.find()
        for document in documents:
            all_documents.append(document)
    except Exception as e:
        raise Exception(f"Fetching documents failed: {e}")
    return all_documents


def load_data(COLLECTION_NAME: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load all documents from a MongoDB collection.
    """
    load_dotenv()

    MONGO_URI = os.getenv("MONGO_URI")
    if MONGO_URI is None:
        raise ValueError("MONGO_URI not found in environment variables.")
    
    DATABASE_NAME = "Film"
    if COLLECTION_NAME is None:
        COLLECTION_NAME = "Data"

    mongo_client = connect_to_mongodb(MONGO_URI)
    database = get_database(mongo_client, DATABASE_NAME)
    collection = get_collection(database, COLLECTION_NAME)
    all_documents = get_all_documents(collection)

    return all_documents


if __name__ == "__main__":
    all_doc = load_data("lsa_svd_preprocessed")
    print(all_doc[:3])
