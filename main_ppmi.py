import os
import pickle
import logging
import time
from dotenv import load_dotenv

from database_connector.qdrant_connector import connect_to_qdrant, insert_points_batch_to_qdrant
from database_connector.mongodb_connector import connect_to_mongodb, get_all_documents, get_collection, get_database
from utils_ppmi import prepare_corpus, create_qdrant_points, get_model_config
from embedding.ppmi import train_ppmi, PPMIEmbedder

# Load .env
load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ENV configs
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
MONGO_SRC_COLLECTION_NAME = os.getenv("LSA_COLLECTION_NAME")

QDRANT_URI = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_KEY")
QDRANT_PPMI_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME")

SAVED_MODEL_PATH = "./embedding/trained_models/ppmi.pkl"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ppmi_model_config.json")


def pipeline(mongo_src_collection, qdrant_client, model: PPMIEmbedder):
    start_time = time.time()

    # Get documents
    logger.info("[pipeline] Getting all documents from MongoDB...")
    all_documents = get_all_documents(mongo_src_collection)
    logger.info(f"[pipeline] Retrieved {len(all_documents)} documents")

    # Prepare corpus
    logger.info("[pipeline] Preparing corpus...")
    corpus = prepare_corpus(all_documents)
    logger.info(f"[pipeline] Prepared corpus with {len(corpus)} items")

    # Transform
    logger.info("[pipeline] Transforming corpus using PPMI model...")
    matrix = model.transform_docs(corpus)

    # Create Qdrant points
    logger.info("[pipeline] Creating Qdrant points...")
    qdrant_points = create_qdrant_points(all_documents, corpus, matrix)
    logger.info(f"[pipeline] Created {len(qdrant_points)} points")

    # Insert to Qdrant
    logger.info(f"[pipeline] Inserting points into Qdrant collection: {QDRANT_PPMI_COLLECTION}")
    try:
        insert_points_batch_to_qdrant(
            qdrant_client=qdrant_client,
            collection_name=QDRANT_PPMI_COLLECTION,
            qdrant_points=qdrant_points
        )
        status = True
    except Exception as e:
        logger.error(f"[pipeline] Failed to upload points: {e}")
        status = False

    if status:
        logger.info("[pipeline] Successfully inserted points into Qdrant")
    else:
        logger.error("[pipeline] Failed to insert points into Qdrant")

    logger.info(f"[pipeline] Total pipeline time: {time.time() - start_time:.2f}s")


def main():
    # Load config
    config = get_model_config(CONFIG_PATH)
    ppmi_args = config['ppmi_args']
    dim_args = config['dim_reduc_args']

    # Connect MongoDB
    logger.info("[main] Connecting to MongoDB...")
    mongo_client = connect_to_mongodb(MONGO_URI)
    mongo_database = get_database(mongo_client, DATABASE_NAME)
    mongo_collection = get_collection(mongo_database, MONGO_SRC_COLLECTION_NAME)

    # Connect Qdrant
    logger.info("[main] Connecting to Qdrant...")
    qdrant_client = connect_to_qdrant(QDRANT_URI, QDRANT_API_KEY)

    # Load or train model
    if os.path.exists(SAVED_MODEL_PATH):
        logger.info(f"[main] Loading existing PPMI model from {SAVED_MODEL_PATH}...")
        with open(SAVED_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        logger.info("[main] PPMI model not found, starting training...")
        all_documents = get_all_documents(mongo_collection)
        corpus = prepare_corpus(all_documents)

        model = train_ppmi(
            docs=corpus,
            max_features=ppmi_args.get("max_features"),
            window_size=ppmi_args.get("window_size"),
            min_count=ppmi_args.get("min_count"),
            n_jobs=ppmi_args.get("n_jobs"),
             n_components=dim_args.get("n_components"),
            save_path=SAVED_MODEL_PATH
        )

    # Run pipeline
    logger.info("[main] Running pipeline...")
    pipeline(mongo_collection, qdrant_client, model)

if __name__ == "__main__":
    main()
