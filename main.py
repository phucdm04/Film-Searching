import os
import pickle
from database_connector.qdrant_connector import connect_to_qdrant, insert_points_batch_to_qdrant
from database_connector.mongodb_connector import connect_to_mongodb, get_all_documents, get_collection, get_database
from utils import prepare_corpus, create_qdrant_points, get_model_config
from embedding.bow_svd_model.final_model import BOW_SVD_Embedding
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# MongoDB config
MONGO_URI=os.getenv("MONGO_URI")
DATABASE_NAME=os.getenv("DATABASE_NAME")
MONGO_SRC_COLLECTION_NAME=os.getenv("LSA_COLLECTION_NAME")

# Qdrant config
QDRANT_URI=os.getenv("QDRANT_URI")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
QDRANT_DEST_COLLECTION_NAME=os.getenv("QDRANT_BOW_SVD_COLLECTION")

# Saved model path
SAVED_MODEL_ROOT = "./embedding/trained_models/"

def pipeline(mongo_src_collection, qdrant_client, model: BOW_SVD_Embedding, save_model: bool=True):
    # Get all data from src_collection
    logger.info('- [pipeline] - Getting all documents from MongoDB collection...')
    all_documents = get_all_documents(mongo_src_collection)
    logger.info(f'- [pipeline] - Got {len(all_documents)} documents')

    # Prepare corpus
    logger.info('- [pipeline] - Preparing corpus...')
    corpus = prepare_corpus(all_documents)
    logger.info(f'- [pipeline] - Prepared {len(corpus)} corpus')

    # Model fit and transform
    logger.info('- [pipeline] - Model fitting and transforming...')
    matrix = model.fit_transform(corpus)

    # Generate points
    logger.info('- [pipeline] - Creating Qdrant points...')
    list_of_points = create_qdrant_points(all_documents, corpus, matrix)
    logger.info(f"- [pipeline] - Created {len(list_of_points)} Qdrant points")

    # Insert to Qdrant collection
    logger.info(f"- [pipeline] - Inserting {len(list_of_points)} points into Qdrant...")
    status = insert_points_batch_to_qdrant(
        qdrant_client=qdrant_client,
        collection_name=QDRANT_DEST_COLLECTION_NAME,
        qdrant_points=list_of_points
    )
    
    # Checking status
    if status:
        logger.info(f"- [pipeline] - Insert {len(list_of_points)} points to Qdrant successfully!")
    else:
        logger.error("- [pipeline] - Failed to insert points to Qdrant.")

    # Save model
    if save_model:
        try:
            os.makedirs(SAVED_MODEL_ROOT, exist_ok=True)
            model_path = os.path.join(SAVED_MODEL_ROOT, f"{BOW_SVD_Embedding.__name__}.pkl")
            logger.info(f"- [pipeline] - Saving model with model path: {model_path}....")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info("- [pipeline] - Successfully saved model!!!")
        except Exception as e:
            logger.error(f"Got an error when saving model: {e}")

def main(model_config):
    # Connect to MongoDB
    logger.info(f"Connecting to MongoDB client...")
    mongo_client = connect_to_mongodb(mongo_uri=MONGO_URI)

    logger.info(f"Getting MongoDB database, database name {DATABASE_NAME}...")
    mongo_database = get_database(
        mongo_client=mongo_client,
        database_name=DATABASE_NAME
    )

    logger.info(f"Getting MongoDB collection, collection name {MONGO_SRC_COLLECTION_NAME}...")
    mongo_src_collection = get_collection(
        database=mongo_database,
        collection_name=MONGO_SRC_COLLECTION_NAME
    )

    # Connect to QdrantDB
    logger.info(f"Connecting to Qdrant VectorDB client...")
    qdrant_client = connect_to_qdrant(
        qdrant_uri=QDRANT_URI,
        api_key=QDRANT_API_KEY
    )

    # Config model
    logger.info(f"Building LSA/SVD model (BOW embedding + SVD dimensionality reduction)...")

    # Log config parameters
    logger.info(f"Model config - BOW args: {model_config.get('bow_args')}")
    logger.info(f"Model config - Dimensionality reduction args: {model_config.get('dim_reduc_args')}")

    model = BOW_SVD_Embedding(
        bow_args=model_config.get('bow_args'),
        dim_reduc_args=model_config.get('dim_reduc_args')
    )

    # Run pipeline
    logger.info(f"Running pipeline...")
    pipeline(
        mongo_src_collection=mongo_src_collection,
        qdrant_client=qdrant_client,
        model=model,
        save_model=True
    )


if __name__ == "__main__":
    model_config = get_model_config()
    main(model_config)