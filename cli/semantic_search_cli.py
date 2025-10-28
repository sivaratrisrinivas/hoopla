#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, verify_model, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    
    single_embed_parser = subparsers.add_parser("embed_text", help="Embed a text")
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings for movie dataset")
    
    query_embed_parser = subparsers.add_parser("embedquery", help="Embed a query text")
    query_embed_parser.add_argument("query", type=str, help="Query to embed")
    
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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()