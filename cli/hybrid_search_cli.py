import argparse

from lib.hybrid_search import normalize_scores, weighted_hybrid_search


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

    weighted_hybrid_parser = subparsers.add_parser(
        "weighted-search",
        help="Run a weighted hybrid search",
    )
    weighted_hybrid_parser.add_argument("query", type=str, help="Search query")
    weighted_hybrid_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting coefficient for hybrid score")
    weighted_hybrid_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_hybrid_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
