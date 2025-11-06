import json
import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import load_movies, load_golden_dataset
from .semantic_search import SemanticSearch

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"


def evaluate_precision(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k


def evaluate_recall(retrieved_docs: list[str], relevant_docs: set[str], k: int = 5) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)

def evaluate_f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0: 
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_command(limit: int = 5):
    movies = load_movies()
    golden_data = load_golden_dataset()
    test_cases = golden_data["test_cases"]

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    total_precision = 0
    results_by_query = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])
        search_results = hybrid_search.rrf_search(query, k=60)
        retrieved_docs = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)
        
        precision = evaluate_precision(retrieved_docs, relevant_docs, limit)
        recall = evaluate_recall(retrieved_docs, relevant_docs, limit)
        f1_score = evaluate_f1_score(precision, recall)
        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "retrieved": retrieved_docs[:limit],
            "relevant": list(relevant_docs),
        }

        total_precision += precision
    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
    }


def llm_judge_results(query: str, results: list[dict]) -> list[int]:
    if not api_key:
        print("GEMINI_API_KEY is not set")
        return [0] * len(results)

    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append(f"{i}. {result['title']}")
    
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    ranking_text = (response.text or "").strip()
    scores = json.loads(ranking_text)

    # The map(int, scores) part converts each score from a string (or float) to an integer.
    # So, if scores = ["2", "1", "3"], after map(int, scores) -> [2, 1, 3].
    # This is needed because the LLM may return numbers as strings or floats, but we want actual integers for evaluation.
    # Returning the mapped list ensures that all results have integer relevance labels, making further processing consistent and correct.
    if len(scores) == len(results):
        return list(map(int, scores))
    
    raise ValueError(f"LLM response parsing error. Expected {len(results)} scores, got {len(scores)}. Response: {scores}")