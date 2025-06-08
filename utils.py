from typing import Dict, List
from numpy.typing import NDArray
from preprocessing.data_models import QdrantPoint
import json

def normalize_text_field(doc: Dict) -> str:
    def normalize_field(field_val):
        return " ".join(w.strip().lower() for w in str(field_val).split(',')) if field_val else ""

    directors = normalize_field(doc['metadata'].get('directors'))
    writers = normalize_field(doc['metadata'].get('writers'))
    description = (doc.get('cleaned_description') or "").lower()

    return f"{directors} {writers} {description}"


def prepare_corpus(documents: List[dict]) -> List[str]:
    # Normalize fields, then merge into one field, call "text"
    corpus = []
    for doc in documents:
        text = normalize_text_field(doc)
        corpus.append(text)
    return corpus

def create_qdrant_points(documents: List[dict], corpus: List[str], embedding_matrix: NDArray) -> List[QdrantPoint]:
    # Check dimension
    if len(documents) != embedding_matrix.shape[0] or embedding_matrix.shape[0] != len(corpus):
        raise ValueError(f"[create_qdrant_points] Shape mismatch: "
                        f"{len(documents)} documents, "
                        f"{embedding_matrix.shape[0]} vectors, "
                        f"{len(corpus)} corpus entries.")
        
     # Create points to insert to QdrantDB, point = vector + metadata + text 
    list_of_points = []
    for idx in range(len(embedding_matrix)):
        point = QdrantPoint(
            text = documents[idx]['original_description'],
            vector = embedding_matrix[idx],
            metadata = documents[idx]['metadata']
        )
        list_of_points.append(point)
    
    return list_of_points

import json

def get_model_config(config_path: str = "bow_svd_model_config.json"):
    """Load BOW + SVD model config from config.json file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    bow_args = config.get('bow_args', {})
    dim_args = config.get('dim_reduc_args', {})

    max_features = bow_args.get('max_features')
    n_components = dim_args.get('n_components')

    # Compute n_components by percent
    if n_components is None and max_features and dim_args.get('n_components_percent'):
        try:
            n_components = int(max_features * dim_args['n_components_percent'])
        except Exception:
            n_components = 100  # fallback default
    elif n_components is None:
        n_components = 100

    dim_reduc_args = {
        'n_components': n_components
    }

    return {
        'bow_args': bow_args,
        'dim_reduc_args': dim_reduc_args
    }
