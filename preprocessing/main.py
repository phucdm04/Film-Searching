from database_connector.mongodb_connector import connect_to_mongodb, get_database, get_collection, get_all_documents
from data_models import RawFilm, CleanFilm, FilmMetadata
from preprocessing.preprocessing import LSASVDPipeline, WordEmbeddingPipeline
import os
from dotenv import load_dotenv
from typing import List
from pydantic import ValidationError
from pymongo import UpdateOne

load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME =  os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Define new collection names
LSA_COLLECTION_NAME = os.getenv("LSA_COLLECTION_NAME")
WEMB_COLLECTION_NAME = os.getenv("WEMB_COLLECTION_NAME")

# Main process
def process_and_upsert(films: List[dict], pipeline, dest_collection_name):
    # Connect to MongoDB client
    mongo_client = connect_to_mongodb(MONGO_URI)
    database = get_database(mongo_client, DATABASE_NAME)
    dest_collection = get_collection(database, dest_collection_name)

    ops = []
    validation_errors = 0

    # Getr all fields in raw data
    required_fields = list(RawFilm.model_fields.keys())

    for raw in films:
        # If raw data missed any fields -> assign None (avoid to loose data)
        for field in required_fields:
            if field not in raw:
                raw[field] = None

        try:
            rf = RawFilm(**raw)
        except ValidationError as e:
            validation_errors += 1
            print(f"Validation error on {raw.get('id')}: {e}")
            continue

        if hasattr(pipeline, "preprocess_single_text"):
            cleaned = pipeline.preprocess_single_text(rf.description)
        elif hasattr(pipeline, "preprocess"):
            cleaned = pipeline.preprocess(rf.description)
        else:
            raise Exception("Invalid pipeline!!!")

        if not cleaned:
            print(f"Warning: cleaned data empty or None for id {rf.id}")
            continue

        cf = CleanFilm(
            id=rf.id,
            original_description=rf.description,
            cleaned_description=cleaned,
            metadata=FilmMetadata(
                film_name=str(rf.film_name),
                image_link=rf.image_link,
                is_adult=rf.isAdult,
                start_year=rf.startYear,
                runtime_minutes=rf.runtimeMinutes,
                genres=rf.genres,
                rating=rf.rating,
                votes=rf.votes,
                directors=rf.directors,
                writers=rf.writers
            )
        )

        ops.append(
            UpdateOne(
                {"id": cf.id},
                {"$set": cf.model_dump(mode="json")},
                upsert=True
            )
        )

    print(f"Validation errors skipped: {validation_errors}")
    if ops:
        result = dest_collection.bulk_write(ops)
        print(f"Matched: {result.matched_count}, Inserted: {result.upserted_count}, Modified: {result.modified_count}")
    else:
        print("No operations to write.")


def main():
    # Connect to MongoDB and get collection
    mongo_client = connect_to_mongodb(MONGO_URI)
    database = get_database(mongo_client, DATABASE_NAME)
    collection = get_collection(database, COLLECTION_NAME)

    # Get all documents for preprocessing
    all_documents = get_all_documents(collection)

    # LSA/SVD pipeline
    lsa = LSASVDPipeline()

    process_and_upsert(
        films=all_documents,
        pipeline=lsa,
        dest_collection_name=LSA_COLLECTION_NAME
    )

    # WordEmbedding pipeline
    wemb = WordEmbeddingPipeline()

    process_and_upsert(
        films=all_documents,
        pipeline=wemb,
        dest_collection_name=WEMB_COLLECTION_NAME
    )
