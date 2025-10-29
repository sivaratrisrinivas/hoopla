# Hoopla

A movie search engine that uses both BM25 ranking and semantic search algorithms to find relevant movies from a large dataset.

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
4. **Optional Chunking**: Quickly split long text into fixed-size word chunks for inspection/debugging

## Quick Start

```bash
# Build the search index (do this first)
python cli/keyword_search_cli.py build

# Basic keyword search
python cli/keyword_search_cli.py search "action thriller"
python cli/keyword_search_cli.py bm25search "romantic comedy" --limit 10

# Semantic search setup and testing
python cli/semantic_search_cli.py verify  # Verify model is loaded
python cli/semantic_search_cli.py verify_embeddings  # Verify embeddings are loaded
python cli/semantic_search_cli.py search "space adventure" --limit 5  # Search movies semantically
# Chunk helper (supports word overlap between chunks)
python cli/semantic_search_cli.py chunk "Long text you want to split..." --chunk-size 200 --overlap 20
# Semantic (sentence) chunking with optional sentence overlap
python cli/semantic_search_cli.py semantic_chunk "Sentence one. Sentence two. Sentence three." --max-chunk-size 2 --overlap 1
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
- `verify_embeddings` - Verify that movie embeddings are loaded and show statistics
- `search <query>` - Search for movies using semantic similarity
- `embed_text <text>` - Generate and display text embeddings (debugging)
- `embedquery <query>` - Generate and display query embeddings (debugging)
- `chunk <text> [--chunk-size <int>] [--overlap <int>]` - Split text into word chunks (default size 200, default overlap 0)
- `semantic_chunk <text> [--max-chunk-size <int>] [--overlap <int>]` - Split text into sentence-based chunks (default max size 4 sentences, default overlap 0)

### Chunking utilities
- **Word chunks**: `chunk` splits words into fixed-size windows; `--overlap` is in words
- **Sentence chunks**: `semantic_chunk` splits on sentence boundaries; `--overlap` is in sentences

## Data

The system searches through a dataset of movies with titles and descriptions. The index and embeddings are cached for fast repeated searches.

## Technical Details

### Keyword Search
- Uses NLTK for text processing and stemming
- Implements BM25 with configurable parameters (k1=1.5, b=0.75)
- Caches processed data for performance
- Supports custom search limits and parameter tuning
- Provides detailed TF-IDF and BM25 scoring for analysis

### Semantic Search
- Uses sentence-transformers library with all-MiniLM-L6-v2 model
- Generates 384-dimensional embeddings for movie descriptions
- Automatically caches embeddings to `cache/movie_embeddings.npy` for fast loading
- Supports text similarity and semantic understanding
- Loads pre-computed embeddings when available, builds them on first run
- Combines movie titles and descriptions for richer semantic understanding
 - Defaults: CPU device; search limit `5` (see `cli/lib/search_utils.py`)
 - Cache directory: `cache/` at project root; embeddings auto-rebuilt if dataset size changes
 - Chunking defaults: `DEFAULT_CHUNK_SIZE=200`, `DEFAULT_CHUNK_OVERLAP=0`, `DEFAULT_MAX_CHUNK_SIZE=4`
