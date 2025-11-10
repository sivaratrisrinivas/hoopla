import os

import numpy as np
from PIL import Image

# Force CPU to avoid CUDA initialization on unsupported GPUs which can crash
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        # Initialize model on CPU explicitly for portability
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]

    
def verify_image_embedding(image_path: str):
    search_instance = MultimodalSearch()
    embedding = search_instance.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")





