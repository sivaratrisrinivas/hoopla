import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    @staticmethod
    def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, 2500)
        semantic_results = self.semantic_search.search_chunks(query, 2500)
        normalized_bm25_scores = normalize_scores([result['score'] for result in bm25_results])
        normalized_semantic_scores = normalize_scores([result['score'] for result in semantic_results])

        doc_scores = {}
        # Collect BM25 scores
        for i, result in enumerate(bm25_results):
            doc_id = result['id']
            doc_scores.setdefault(doc_id, {
                "doc": {"id": result['id'], "title": result['title'], "description": result.get('document', '')},
                "bm25": 0.0,
                "semantic": 0.0,
            })
            doc_scores[doc_id]["bm25"] = normalized_bm25_scores[i]

        # Collect semantic scores
        for i, result in enumerate(semantic_results):
            doc_id = result['id']
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": {"id": result['id'], "title": result['title'], "description": result.get('document', '')},
                    "bm25": 0.0,
                    "semantic": 0.0,
                }
            doc_scores[doc_id]["semantic"] = normalized_semantic_scores[i]

        # Calculate hybrid score for each doc and sort descending by hybrid score
        for doc_id, entry in doc_scores.items():
            bm25 = entry["bm25"]
            semantic = entry["semantic"]
            entry["hybrid"] = self.hybrid_score(bm25, semantic, alpha)

        # Convert to list of dicts
        results = []
        for doc_id, entry in doc_scores.items():
            results.append({
                "id": doc_id,
                "doc": entry["doc"],
                "bm25": entry["bm25"],
                "semantic": entry["semantic"],
                "hybrid": entry["hybrid"],
            })

        # Sort by hybrid score descending and return up to `limit`
        results.sort(key=lambda x: x["hybrid"], reverse=True)
        return results[:limit]



    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores using min-max normalization to range [0, 1]."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    # If all scores are the same, return all 1.0
    if min_score == max_score:
        return [1.0] * len(scores)
    
    # Min-max normalization: (score - min) / (max - min)
    normalized_scores = []
    for score in scores:
        normalized_scores.append((score - min_score) / (max_score - min_score))

    return normalized_scores

def weighted_hybrid_search(query: str, alpha: float, limit: int = 10) -> list[dict]:
    """
    Runs a weighted hybrid search using the instance's weighted_search method and returns formatted results.

    Args:
        query (str): Search query.
        alpha (float): Weighting coefficient for hybrid score.
        limit (int): Number of results to return.

    Returns:
        list[dict]: List of result dictionaries in sorted order.
    """
    searcher = HybridSearch(load_movies())
    results = searcher.weighted_search(query, alpha, limit)
    for result in results:
        print(result['doc']['title'])
    return results