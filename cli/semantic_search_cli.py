#!/usr/bin/env python3

import argparse

from lib.search_utils import load_movies
from lib.semantic_search import embed_text, verify_model, verify_embeddings, embed_query_text, semantic_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    
    single_embed_parser = subparsers.add_parser("embed_text", help="Embed a text")
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings for movie dataset")
    
    query_embed_parser = subparsers.add_parser("embedquery", help="Embed a query text")
    query_embed_parser.add_argument("query", type=str, help="Query to embed")
    
    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            semantic_search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()