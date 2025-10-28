# Hoopla

A movie search engine that uses advanced text search algorithms to find relevant movies from a large dataset.

## What

Hoopla is a command-line tool that searches through thousands of movie descriptions using both BM25 ranking and semantic search algorithms. It can find movies based on keywords, themes, or any text you search for using traditional keyword matching or AI-powered semantic understanding.

## Why

Traditional keyword matching just looks for exact word matches. BM25 is smarter - it considers how often words appear, how rare they are across all movies, and adjusts scores based on document length. Semantic search goes even further by understanding the meaning and context of your search, finding movies that are conceptually similar even if they don't share exact keywords.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd hoopla

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Data Setup

Place your movie dataset as `data/movies.json` with the following structure:
```json
{
  "movies": [
    {
      "id": 1,
      "title": "Movie Title",
      "description": "Movie description..."
    }
  ]
}
```

Also ensure `data/stopwords.txt` exists with common stopwords (one per line).

## How

The system works with two complementary search approaches:

### Keyword Search (BM25)
1. **Index Building**: Processes all movie data and creates an inverted index (like a book's index, but for every word)
2. **Text Processing**: Cleans and stems words (removes endings like "running" → "run") 
3. **Smart Scoring**: Uses BM25 algorithm to rank results by relevance

### Semantic Search
1. **Model Loading**: Loads a pre-trained sentence transformer model (all-MiniLM-L6-v2)
2. **Text Embedding**: Converts search queries and movie descriptions into high-dimensional vectors
3. **Similarity Matching**: Finds movies with similar semantic meaning using cosine similarity

## Quick Start

```bash
# Build the search index (do this first)
python cli/keyword_search_cli.py build

# Basic keyword search
python cli/keyword_search_cli.py search "action thriller"
python cli/keyword_search_cli.py bm25search "romantic comedy" --limit 10

# Semantic search setup and testing
python cli/semantic_search_cli.py verify  # Verify model is loaded
python cli/semantic_search_cli.py embed_text "Luke, I am your father"  # Test embedding
python cli/semantic_search_cli.py verify_embeddings  # Verify embeddings are loaded
python cli/semantic_search_cli.py embedquery "space adventure"  # Test query embedding
```

## Available Commands

### Keyword Search Commands
- `build` - Create the search index from movie data
- `search <query>` - Basic keyword search (exact token matching)
- `bm25search <query>` - Search movies with BM25 scoring
- `tf <doc_id> <term>` - Get term frequency for a word in a movie
- `idf <term>` - Get inverse document frequency for a word
- `tfidf <doc_id> <term>` - Get TF-IDF score
- `bm25idf <term>` - Get BM25-IDF score
- `bm25tf <doc_id> <term>` - Get BM25 term frequency score

### Semantic Search Commands
- `verify` - Verify that the embedding model is loaded correctly
- `embed_text <text>` - Generate and display text embeddings
- `embedquery <query>` - Generate and display query embeddings
- `verify_embeddings` - Verify that movie embeddings are loaded and show statistics

## Data

The system searches through a dataset of movies with titles and descriptions. The index and embeddings are cached for fast repeated searches.

## Technical Details

### Keyword Search
- Uses NLTK for text processing and stemming
- Implements BM25 with configurable parameters (k1=1.5, b=0.75)
- Caches processed data for performance
- Supports custom search limits and parameter tuning

### Semantic Search
- Uses sentence-transformers library with all-MiniLM-L6-v2 model
- Generates 384-dimensional embeddings for movie descriptions
- Automatically caches embeddings to `cache/movie_embeddings.npy` for fast loading
- Supports text similarity and semantic understanding
- Loads pre-computed embeddings when available, builds them on first run
