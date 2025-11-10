import os
from unittest import result

import numpy as np
from PIL import Image

# Force CPU to avoid CUDA initialization on unsupported GPUs which can crash
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sentence_transformers import SentenceTransformer

from .search_utils import load_movies, format_search_result
from .semantic_search import cosine_similarity
 

class MultimodalSearch:
    def __init__(self, documents = [], model_name="clip-ViT-B-32"):
        # Initialize model on CPU explicitly for portability
        self.model = SentenceTransformer(model_name, device="cpu")
        self.documents = documents
        self.texts = []
        for doc in self.documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar = True)

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[dict]:    
        image_embedding = self.embed_image(image_path)
        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:limit]:
            doc = self.documents[idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"],
                    score=score,
                )
            )
        return results


def image_search_command(image_path="data/paddington.jpeg", limit=5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    movies = load_movies()
    search_instance = MultimodalSearch(movies)
    results = search_instance.search_with_image(image_path, limit)
    return {"image_path": image_path, "results": results}


    
def verify_image_embedding(image_path: str):
    search_instance = MultimodalSearch()
    embedding = search_instance.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")





