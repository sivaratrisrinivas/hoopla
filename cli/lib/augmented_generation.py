import os
from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import DEFAULT_SEARCH_LIMIT, RRF_K, SEARCH_MULTIPLIER, load_movies

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def generate_answer(search_results: list[dict], query: str, limit: 5):
    context = ""
    for result in search_results:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{context}

Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(
        model=model, 
        contents=prompt,
    )
    return (response.text or "").strip()



def rag(query: str, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }
    
    answer = generate_answer(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results,
        "answer": answer,
    }

def rag_command(query):
    return rag(query)

    