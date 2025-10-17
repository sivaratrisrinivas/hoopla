#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command, build_command, InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get the term frequency of a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            idx = InvertedIndex()
            idx.load()
            tf = idx.get_tf(args.doc_id, args.term)
            print(f"{tf}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()