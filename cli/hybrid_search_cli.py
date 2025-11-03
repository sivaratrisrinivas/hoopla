import argparse
import os
from dotenv import load_dotenv
from google import genai

from lib.hybrid_search import normalize_scores, weighted_search_command, rrf_search_command

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize list of scores using min-max normalization",
    )
    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="List of scores to normalize",
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Perform RRF hybrid search"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "--k", type=int, default=60, help="Constant for RRF (default=60)"
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )
    rrf_parser.add_argument(
    "--enhance",
    type=str,
    choices=["spell"],
    help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case "rrf-search":
            # INSERT_YOUR_CODE
            enhanced_query = args.query
            if getattr(args, "enhance", None) == "spell":
                try:
                    api_key = os.environ.get("GEMINI_API_KEY")
                    if not api_key:
                        raise ValueError("GEMINI_API_KEY not set in environment variables.")

                    client = genai.Client(api_key=api_key)
                    system_prompt = (
                        f"Fix any spelling errors in this movie search query.\n\n"
                        f"Only correct obvious typos. Don't change correctly spelled words.\n\n"
                        f'Query: "{args.query}"\n\n'
                        f"If no errors, return the original query.\n"
                        f"Corrected:"
                    )

                    response = client.models.generate_content(
                        model="gemini-2.0-flash-001",
                        contents=system_prompt,
                    )
                    # Use stripped response for enhanced query
                    enhanced_query_candidate = response.text.strip()
                    # Gemini may echo the prompt, try to extract just the actual correction
                    if enhanced_query_candidate.lower().startswith("corrected:"):
                        enhanced_query_candidate = enhanced_query_candidate[len("corrected:"):].strip()
                    
                    # Strip surrounding quotes if present
                    if enhanced_query_candidate.startswith('"') and enhanced_query_candidate.endswith('"'):
                        enhanced_query_candidate = enhanced_query_candidate[1:-1]
                    elif enhanced_query_candidate.startswith("'") and enhanced_query_candidate.endswith("'"):
                        enhanced_query_candidate = enhanced_query_candidate[1:-1]

                    enhanced_query = enhanced_query_candidate
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                except Exception as e:
                    print(f"[Warning] Query enhancement failed: {e}")
                    enhanced_query = args.query
            result = rrf_search_command(enhanced_query, args.k, args.limit)

            print(
                f"RRF Hybrid Search Results for '{result['query']}' (k={result['k']}):"
            )
            print(f"  K {result['k']}: {int(result['k'] * 100)}% BM25, {int((1 - result['k']) * 100)}% Semantic")
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
