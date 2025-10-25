from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
    
    def generate_embedding(self, text: str) -> list[float]:
        if text is None or text.strip() == "":
            raise ValueError("Text cannot be empty")
        embedding = self.model.encode([text])
        return embedding[0]


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    # for i, dim in enumerate(embedding[:3]):
    print(f"Embedding: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")



def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")
