#!/usr/bin/env python3

import argparse
import os
import pickle
from lib.keyword_search import search_command, tokenize_text
from lib.search_utils import load_movies


class InvertedIndex: 
    def __init__(self):
        self.index = {}
        self.docmap = {}
    
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        for token in tokens: 
            if token not in self.index: 
                self.index[token] = []
            self.index[token].append(doc_id)
    
    def get_documents(self, term):
        term = term.lower()
        doc_ids = self.index.get(term, [])
        return sorted(set(doc_ids))

    def build(self):
        movies = load_movies()
        for idx, m in enumerate(movies):
            text = f"{m['title']} {m['description']}"
            self.__add_document(idx + 1, text)  # Use 1-based indexing
            self.docmap[idx + 1] = m  # Use 1-based indexing
    
    def save(self):        
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f_index:
            pickle.dump(self.index, f_index)
        with open("cache/docmap.pkl", "wb") as f_docmap:
            pickle.dump(self.docmap, f_docmap)

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    if docs:
        print(f"First document for token 'merida' = {docs[0]}")
    else:
        print("No documents found for token 'merida'.")

# Add "build" command to CLI
# Mention: a new "build" command is added to the CLI, and main() is updated to support it.

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # Mention: add "build" subcommand
    build_parser = subparsers.add_parser("build", help="Build the inverted index and cache data")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            # Mention: run build_command() when "build" is specified
            build_command()
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()