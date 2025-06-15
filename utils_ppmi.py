from typing import Dict, List
from numpy.typing import NDArray
from preprocessing.data_models import QdrantPoint
import json
import re


def normalize_token(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_field(field_val):
    if not field_val:
        return ""
    words = str(field_val).split(',')
    normalized = [normalize_token(w) for w in words]
    return " ".join(normalized)

def normalize_text_field(doc: Dict) -> str:
    directors = normalize_field(doc['metadata'].get('directors'))
    writers = normalize_field(doc['metadata'].get('writers'))
    genres = normalize_field(doc['metadata'].get('genres'))
    actors = normalize_field(doc['metadata'].get('actors'))
    description = normalize_token(doc.get('cleaned_description') or "")

    return f"{directors} {writers} {genres} {actors} {description}".strip()

def prepare_corpus(documents: List[dict]) -> List[str]:
    return [normalize_text_field(doc) for doc in documents]

def create_qdrant_points(documents: List[dict], corpus: List[str], embedding_matrix: NDArray) -> List[QdrantPoint]:
    if len(documents) != embedding_matrix.shape[0] or embedding_matrix.shape[0] != len(corpus):
        raise ValueError(f"[create_qdrant_points] Shape mismatch: "
                         f"{len(documents)} documents, "
                         f"{embedding_matrix.shape[0]} vectors, "
                         f"{len(corpus)} corpus entries.")

    list_of_points = []
    for idx in range(len(embedding_matrix)):
        point = QdrantPoint(
            text=documents[idx]['original_description'],
            vector=embedding_matrix[idx],
            metadata=documents[idx]['metadata']
        )
        list_of_points.append(point)

    return list_of_points

def get_model_config(config_path: str = "ppmi_model_config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)

    ppmi_args = {
        'max_features': config['ppmi_args'].get('max_features'),
        'window_size': config['ppmi_args'].get('window_size', 4),
        'min_count': config['ppmi_args'].get('min_count', 2),
        'n_jobs': config['ppmi_args'].get('n_jobs', 1)
    }

    n_components_percent = config['dim_reduc_args'].get('n_components_percent', 0.95)
    max_features = ppmi_args['max_features']
    n_components = int(max_features * n_components_percent)

    return {
        'ppmi_args': ppmi_args,
        'dim_reduc_args': {
            'n_components': n_components
        }
    }
