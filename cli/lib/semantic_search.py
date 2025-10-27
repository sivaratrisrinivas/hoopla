import os
from .search_utils import load_movies
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        
        # Create string representations of all documents
        document_strings = []
        for document in documents:
            document_string = f"{document['title']}: {document['description']}"
            document_strings.append(document_string)
        
        # Encode all documents at once
        self.embeddings = self.model.encode(document_strings, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {document["id"]: document for document in documents}
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)



    
    def generate_embedding(self, text: str) -> list[float]:
        if text is None or text.strip() == "":
            raise ValueError("Cannot generate embedding for empty text")
        embedding = self.model.encode([text])
        return embedding[0]


def embed_text(text: str):
    semantic_search_instance = SemanticSearch()
    embedding = semantic_search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"Embedding: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")



def verify_model():
    semantic_search_instance = SemanticSearch()
    print(f"Model loaded: {semantic_search_instance.model}")
    print(f"Max sequence length: {semantic_search_instance.model.max_seq_length}")


def verify_embeddings():
    semantic_search_instance = SemanticSearch()
    movies = load_movies()
    semantic_search_instance.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(semantic_search_instance.documents)}")
    print(f"Embeddings shape: {semantic_search_instance.embeddings.shape[0]} vectors in {semantic_search_instance.embeddings.shape[1]} dimensions")

