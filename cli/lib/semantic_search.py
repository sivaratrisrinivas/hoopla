import os
import numpy as np

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies

from sentence_transformers import SentenceTransformer

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for document in documents:
            self.document_map[document["id"]] = document
            movie_strings.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        
        
        query_embedding = self.generate_embedding(query)
        
        # Calculate similarity scores for all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity_score, self.documents[i]))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Prepare the output list of dicts with the required fields
        results = []
        for score, doc in similarities[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })
        return results

    
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
    documents = load_movies()
    embeddings = semantic_search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    semantic_search_instance = SemanticSearch()
    embedding = semantic_search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit = DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


