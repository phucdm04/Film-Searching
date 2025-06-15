import sys
import os

import numpy as np
from embedding.glove import GloVe, get_sentences
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from preprocessing.preprocessing import LSASVDPipeline
from qdrant_connector import insert_point_to_qdrant
import uuid

load_dotenv()

def upsert_descriptions_to_qdrant(model: GloVe, sentences: list[str], client: QdrantClient, collection_name: str):
    """Encode and insert each sentence to Qdrant using insert_point_to_qdrant."""
    for text in sentences:
        vec = model.encode(text)
        if np.linalg.norm(vec) == 0:
            continue
        
        qdrant_point = {
            "id": str(uuid.uuid4()),
            "vector": vec,
            "text": text,
            "metadata": {"source": "glove_upsert"}
        }

        success = insert_point_to_qdrant(client, collection_name, qdrant_point)
        if not success:
            print(f"⚠️ Failed to insert: {text[:50]}...")

if __name__ == "__main__":
    glove_model = GloVe.from_pretrained("./trained_models/glove.pkl")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_KEY"),
    )
    collection_name = "glove"

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=glove_model.embedding_dim,
            distance=Distance.COSINE,
        )
    )

    sentences = get_sentences()
    print("Model vocab size:", glove_model.vocab_size)
    print("Total sentences:", len(sentences))

    upsert_descriptions_to_qdrant(glove_model, sentences, client, collection_name)

    print("✅ Done upserting to Qdrant.")