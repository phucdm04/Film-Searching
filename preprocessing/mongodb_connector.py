import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME =  os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def connect_to_mongodb(mongo_uri):
    """Function to connect to mongodb client.
    Return mongodb client if successfull."""

    # Connect to server
    mongo_client = MongoClient(mongo_uri)

    # Ping to server
    try:
        mongo_client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        raise Exception(f"Connection failed: {e}")

    return mongo_client

def get_database(mongo_client, database_name):
    """Function to get database from mongodb client."
    Return database if successfull."""

    # Get database
    try:
        database = mongo_client[database_name]
        print(f"Database '{database_name}' connected successfully!")
    except Exception as e:
        raise Exception(f"Connection failed: {e}")

    return database

def get_collection(database, collection_name):
    """Function to get collection from database."
    Return collection if successfull."""

    # Get collection
    try:
        collection = database[collection_name]
        print(f"Collection '{collection_name}' connected successfully!")
    except Exception as e:
        raise Exception(f"Connection failed: {e}")

    return collection

def get_all_documents(collection):
    """Function to get all documents from collection.
        Return list of documents.
    """
    all_documents = []
    try:
        documents = collection.find()
        for document in documents:
            all_documents.append(document)
    except Exception as e:
        raise Exception(f"Connection failed: {e}")

    return all_documents

def get_documents_by_index(collection, start_index, end_index):
    """Function to get documents from collection by index.
        Return list of documents.
    """
    documents = []
    try:
        documents = collection.find().skip(start_index).limit(end_index)
    except Exception as e:
        raise Exception(f"Connection failed: {e}")

    return documents