import argparse

from lib.hybrid_search import hybrid_search
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    hybrid_search_parser = subparsers.add_parser(
        "hybrid_search",
        help="Search for movies using hybrid search",
    )
    hybrid_search_parser.add_argument("query", type=str, help="Search query")
    hybrid_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )


    args = parser.parse_args()

    match args.command:
        case "hybrid_search":
            result = hybrid_search(args.query, args.limit)
            print(f"Query: {result['query']}")
            print("Results:")
            for i, res in enumerate(result['results']):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()