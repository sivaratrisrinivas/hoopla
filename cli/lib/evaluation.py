import json
import math
import os
import platform
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .keyword_search import InvertedIndex
from .reranking import rerank
from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_golden_dataset,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
model = "gemini-2.0-flash-001"
_client = None

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
CONFIG_ORDER = ("bm25", "semantic", "rrf", "rrf_cross_encoder")
CONFIG_LABELS = {
    "bm25": "BM25",
    "semantic": "Semantic",
    "rrf": "RRF",
    "rrf_cross_encoder": "RRF + cross-encoder",
}


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


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


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def _titles(search_results: list[dict], limit: int) -> list[str]:
    retrieved = []
    seen = set()
    for result in search_results:
        title = result.get("title", "")
        if not title or title in seen:
            continue
        seen.add(title)
        retrieved.append(title)
        if len(retrieved) >= limit:
            break
    return retrieved


def _retrieval_cache_paths() -> list[str]:
    idx = InvertedIndex()
    return [
        idx.index_path,
        idx.docmap_path,
        idx.term_frequencies_path,
        idx.doc_lengths_path,
        CHUNK_EMBEDDINGS_PATH,
        CHUNK_METADATA_PATH,
    ]


def _clear_retrieval_cache() -> None:
    for path in _retrieval_cache_paths():
        if os.path.exists(path):
            os.remove(path)


def _hybrid_search_from_built(
    movies: list[dict],
    semantic_search: ChunkedSemanticSearch,
    idx: InvertedIndex,
) -> HybridSearch:
    # Skip HybridSearch.__init__ so the timed objects are the retriever we score.
    hybrid_search = HybridSearch.__new__(HybridSearch)
    hybrid_search.documents = movies
    hybrid_search.semantic_search = semantic_search
    hybrid_search.idx = idx
    return hybrid_search


def _build_scored_hybrid_search(
    movies: list[dict],
) -> tuple[HybridSearch, float, float]:
    """Cold-build the HybridSearch instance that evaluate_command will score."""
    _clear_retrieval_cache()

    semantic_search = ChunkedSemanticSearch()
    chunk_started = time.perf_counter()
    semantic_search.build_chunk_embeddings(movies)
    chunk_seconds = time.perf_counter() - chunk_started

    idx = InvertedIndex()
    bm25_started = time.perf_counter()
    idx.build()
    idx.save()
    bm25_seconds = time.perf_counter() - bm25_started

    hybrid_search = _hybrid_search_from_built(movies, semantic_search, idx)
    return hybrid_search, bm25_seconds, chunk_seconds


def _hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["memory_gb"] = round(kb / 1024 / 1024, 2)
                    break
    except OSError:
        pass
    return info


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _score_query(retrieved: list[str], relevant: set[str], k: int) -> dict:
    precision = evaluate_precision(retrieved, relevant, k)
    recall = evaluate_recall(retrieved, relevant, k)
    f1_score = evaluate_f1_score(precision, recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "retrieved": retrieved[:k],
        "relevant": sorted(relevant),
    }


def evaluate_command(limit: int = 5) -> dict:
    movies = load_movies()
    golden_data = load_golden_dataset()
    test_cases = golden_data["test_cases"]
    movie_titles = {movie["title"] for movie in movies}
    missing_relevant = sorted(
        {
            title
            for case in test_cases
            for title in case["relevant_docs"]
            if title not in movie_titles
        }
    )

    hybrid_search, bm25_build_seconds, chunk_embed_seconds = (
        _build_scored_hybrid_search(movies)
    )

    def retrieve_bm25(query: str) -> list[dict]:
        return hybrid_search._bm25_search(query, limit)

    def retrieve_semantic(query: str) -> list[dict]:
        return hybrid_search.semantic_search.search_chunks(query, limit)

    def retrieve_rrf(query: str) -> list[dict]:
        return hybrid_search.rrf_search(query, k=RRF_K, limit=limit)

    def retrieve_rrf_ce(query: str) -> list[dict]:
        search_limit = limit * SEARCH_MULTIPLIER
        fused = hybrid_search.rrf_search(query, k=RRF_K, limit=search_limit)
        return rerank(query, fused, method="cross_encoder", limit=limit)

    retrievers = {
        "bm25": retrieve_bm25,
        "semantic": retrieve_semantic,
        "rrf": retrieve_rrf,
        "rrf_cross_encoder": retrieve_rrf_ce,
    }

    configurations = {}
    for config_name in CONFIG_ORDER:
        retriever = retrievers[config_name]
        per_query = {}
        latencies_ms = []
        precisions = []
        recalls = []
        f1s = []
        failures = []
        for test_case in test_cases:
            query = test_case["query"]
            relevant_docs = set(test_case["relevant_docs"])
            try:
                started = time.perf_counter()
                search_results = retriever(query)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                retrieved_docs = _titles(search_results, limit)
                scored = _score_query(retrieved_docs, relevant_docs, limit)
                scored["latency_ms"] = elapsed_ms
                per_query[query] = scored
                latencies_ms.append(elapsed_ms)
                precisions.append(scored["precision"])
                recalls.append(scored["recall"])
                f1s.append(scored["f1_score"])
            except Exception as exc:
                failures.append({"query": query, "error": f"{type(exc).__name__}: {exc}"})
                per_query[query] = {
                    "precision": None,
                    "recall": None,
                    "f1_score": None,
                    "retrieved": [],
                    "relevant": sorted(relevant_docs),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        configurations[config_name] = {
            "label": CONFIG_LABELS[config_name],
            "precision_at_k": _mean(precisions),
            "recall_at_k": _mean(recalls),
            "f1": _mean(f1s),
            "p95_query_latency_ms": percentile(latencies_ms, 95),
            "mean_query_latency_ms": _mean(latencies_ms),
            "queries_evaluated": len(precisions),
            "failures": failures,
            "per_query": per_query,
        }

    report = {
        "schema": "hoopla-gs-t3-retrieval-v1",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "k": limit,
        "rrf_k": RRF_K,
        "search_multiplier": SEARCH_MULTIPLIER,
        "models": {
            "embedding": EMBEDDING_MODEL,
            "cross_encoder": CROSS_ENCODER_MODEL,
        },
        "dataset": {
            "path": "data/golden_dataset.json",
            "num_queries": len(test_cases),
            "num_documents": len(movies),
            "missing_relevant_titles": missing_relevant,
        },
        "hardware": _hardware_info(),
        "index_build": {
            "bm25_seconds": bm25_build_seconds,
            "chunk_embeddings_seconds": chunk_embed_seconds,
            "total_seconds": bm25_build_seconds + chunk_embed_seconds,
            "cache_hit": False,
            "retriever": "HybridSearch",
        },
        "configurations": configurations,
        # Preserve the original RRF-only shape for callers that still read it.
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": configurations["rrf"]["per_query"],
    }
    return report


def format_markdown_table(report: dict) -> str:
    k = report["k"]
    lines = [
        f"| Configuration | Precision@{k} | Recall@{k} | F1 | p95 latency (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CONFIG_ORDER:
        cfg = report["configurations"][name]
        if cfg["precision_at_k"] is None:
            lines.append(
                f"| {cfg['label']} | FAILED | FAILED | FAILED | FAILED |"
            )
            continue
        p95 = cfg["p95_query_latency_ms"]
        lines.append(
            f"| {cfg['label']} | {cfg['precision_at_k']:.4f} | {cfg['recall_at_k']:.4f} | {cfg['f1']:.4f} | {p95:.1f} |"
        )
    build = report["index_build"]
    lines.append("")
    cache_label = "cache hit" if build.get("cache_hit") else "cold rebuild"
    lines.append(
        f"Index build time: {build['total_seconds']:.2f}s "
        f"(BM25 {build['bm25_seconds']:.2f}s, chunk embeddings {build['chunk_embeddings_seconds']:.2f}s, {cache_label})."
    )
    missing = report["dataset"].get("missing_relevant_titles") or []
    if missing:
        lines.append(
            "Golden-set titles absent from movies.json: " + ", ".join(missing) + "."
        )
    failures = []
    for name in CONFIG_ORDER:
        failures.extend(
            f"{CONFIG_LABELS[name]} / {item['query']}: {item['error']}"
            for item in report["configurations"][name]["failures"]
        )
    if failures:
        lines.append("Failures:")
        for item in failures:
            lines.append(f"- {item}")
    return "\n".join(lines)


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

    response = _get_client().models.generate_content(
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
