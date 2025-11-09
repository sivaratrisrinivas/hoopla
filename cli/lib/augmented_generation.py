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


def generate_summary(search_results: list[dict], query: str, limit: 5):
    context = ""
    for result in search_results:
        context += f"{result['title']}: {result['document']}\n\n"
    
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{context}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (response.text or "").strip()

def summarize(query: str, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }
    
    summary = generate_summary(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results,
        "summary": summary,
    }


def summarize_command(query: str, limit=DEFAULT_SEARCH_LIMIT):
    return summarize(query, limit)
    

def generate_summary_with_citations(search_results: list[dict], query: str, limit: 5):
    context = ""
    for result in search_results:
        context += f"{result['title']}: {result['document']}\n\n"
    
    prompt = prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{context}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return (response.text or "").strip()


def summarize_with_citations(query: str, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    summary = generate_summary_with_citations(search_results, query, limit)

    return {
        "query": query,
        "search_results": search_results,
        "summary": summary,
    }


def summarize_with_citations_command(query: str, limit=DEFAULT_SEARCH_LIMIT):
    return summarize_with_citations(query, limit)