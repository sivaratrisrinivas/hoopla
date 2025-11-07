import argparse

from lib.hybrid_search import rrf_search_command
from lib.augmented_generation import rag

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query)
            docs = [result["document"] for result in results["results"]]
            answer = rag(query, docs)
            print("Search Results:\n")
            for doc in docs:
                print(f"  - {doc}\n")
            print("RAG Response:\n")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()