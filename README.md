# Hoopla

A movie search engine that uses advanced text search algorithms to find relevant movies from a large dataset.

## What

Hoopla is a command-line tool that searches through thousands of movie descriptions using the BM25 ranking algorithm. It can find movies based on keywords, themes, or any text you search for.

## Why

Traditional keyword matching just looks for exact word matches. BM25 is smarter - it considers how often words appear, how rare they are across all movies, and adjusts scores based on document length. This gives you much better search results that actually match what you're looking for.

## How

The system works in three steps:

1. **Index Building**: Processes all movie data and creates an inverted index (like a book's index, but for every word)
2. **Text Processing**: Cleans and stems words (removes endings like "running" → "run") 
3. **Smart Scoring**: Uses BM25 algorithm to rank results by relevance

## Quick Start

```bash
# Build the search index (do this first)
python cli/keyword_search_cli.py build

# Search for movies
python cli/keyword_search_cli.py bm25search "action thriller"
python cli/keyword_search_cli.py bm25search "romantic comedy" --limit 10
```

## Available Commands

- `build` - Create the search index from movie data
- `bm25search <query>` - Search movies with BM25 scoring
- `tf <doc_id> <term>` - Get term frequency for a word in a movie
- `idf <term>` - Get inverse document frequency for a word
- `tfidf <doc_id> <term>` - Get TF-IDF score
- `bm25idf <term>` - Get BM25-IDF score
- `bm25tf <doc_id> <term>` - Get BM25 term frequency score

## Data

The system searches through a dataset of movies with titles and descriptions. The index is cached for fast repeated searches.

## Technical Details

- Uses NLTK for text processing and stemming
- Implements BM25 with configurable parameters (k1=1.5, b=0.75)
- Caches processed data for performance
- Supports custom search limits and parameter tuning
