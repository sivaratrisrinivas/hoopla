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

The system works with two complementary search approaches that can be combined:

### Keyword Search (BM25)
1. **Index Building**: Processes all movie data and creates an inverted index (like a book's index, but for every word)
2. **Text Processing**: Cleans and stems words (removes endings like "running" → "run") 
3. **Smart Scoring**: Uses BM25 algorithm to rank results by relevance

### Semantic Search
1. **Model Loading**: Loads a pre-trained sentence transformer model (all-MiniLM-L6-v2)
2. **Text Embedding**: Converts search queries and movie descriptions into high-dimensional vectors
3. **Similarity Matching**: Finds movies with similar semantic meaning using cosine similarity
4. **Optional Chunking**: Quickly split long text into fixed-size word chunks for inspection/debugging
5. **Chunked Embeddings**: Precompute sentence-chunk embeddings for the dataset for faster semantic ops

### Hybrid Search
1. **Dual Query**: Runs both BM25 and semantic search on the same query
2. **Result Merging**: Combines results from both approaches into a single result set
3. **Score-Based Ranking**: Sorts merged results by relevance score and returns top matches

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

# Precompute and use chunked semantic search (sentence-level)
python cli/semantic_search_cli.py embed_chunks
python cli/semantic_search_cli.py search_chunked "superhero action movie" --limit 10

# Hybrid search (combines BM25 and semantic search)
python cli/hybrid_search_cli.py hybrid_search "superhero action movie" --limit 10

# Weighted hybrid search with configurable alpha (0.0 = pure semantic, 1.0 = pure BM25)
python cli/hybrid_search_cli.py weighted-search "British Bear" --alpha 0.5 --limit 25

# RRF (Reciprocal Rank Fusion) hybrid search with configurable k constant
python cli/hybrid_search_cli.py rrf-search "British Bear" --k 60 --limit 25

# Normalize scores using min-max normalization
python cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1
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
- `embed_chunks` - Generate and cache sentence-chunk embeddings for all movies
- `search_chunked <query> [--limit <int>]` - Search using chunked embeddings aggregated to movie level
- `chunk <text> [--chunk-size <int>] [--overlap <int>]` - Split text into word chunks (default size 200, default overlap 0)
- `semantic_chunk <text> [--max-chunk-size <int>] [--overlap <int>]` - Split text into sentence-based chunks (default max size 4 sentences, default overlap 0)

### Hybrid Search Commands
- `hybrid_search <query> [--limit <int>]` - Search for movies using combined BM25 and semantic search results
- `weighted-search <query> [--alpha <float>] [--limit <int>]` - Weighted hybrid search with configurable alpha coefficient (default 0.5)
- `rrf-search <query> [--k <int>] [--limit <int>]` - Reciprocal Rank Fusion (RRF) hybrid search with configurable k constant (default 60)
- `normalize <scores...>` - Normalize scores using min-max normalization to range [0, 1]

### Chunking utilities
- **Word chunks**: `chunk` splits words into fixed-size windows; `--overlap` is in words
- **Sentence chunks**: `semantic_chunk` splits on sentence boundaries; `--overlap` is in sentences
  - Handles edge cases: strips whitespace from input and sentences, filters empty chunks, handles text without punctuation
  - Test edge cases: `" Leading and trailing spaces. "`, `"Text without punctuation"`, `" "`, `""`

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
 - Chunking defaults: `DEFAULT_CHUNK_SIZE=200`, `DEFAULT_CHUNK_OVERLAP=1`, `DEFAULT_SEMANTIC_CHUNK_SIZE=4`
 - Chunked search: computes cosine similarity per chunk, keeps best chunk score per movie, sorts desc, returns top `--limit`. Caches chunk vectors at `cache/chunk_embeddings.npy` and metadata at `cache/chunk_metadata.json`.
 - Semantic chunking handles edge cases: strips leading/trailing whitespace, filters empty sentences and chunks, treats single non-punctuated sentences as complete chunks.

### Precomputing chunked embeddings
- Run to create sentence-chunk embeddings and metadata cache:

```bash
python cli/semantic_search_cli.py embed_chunks
# Expected output example: "Generated 72909 chunked embeddings"
```

**Note**: If you modify the chunking logic, delete the cache files to force a rebuild:
```bash
rm cache/chunk_embeddings.npy cache/chunk_metadata.json
```
Then run `embed_chunks` again to regenerate embeddings with the updated chunking.

### Chunked semantic search output format
Results are printed as:

```
1. The Incredibles (score: 0.8123)
   A family of undercover superheroes, while trying to live the quiet suburban life...
```

### Hybrid Search
- Combines BM25 keyword search and chunked semantic search results
- Merges results from both approaches, sorts by score, and returns top `--limit` results
- Requires both keyword index (run `keyword_search_cli.py build`) and semantic chunk embeddings (run `semantic_search_cli.py embed_chunks`)
- Output format matches chunked semantic search with title, score, and description preview
- **Weighted Hybrid Search**: `weighted-search` command combines normalized BM25 and semantic scores using weighted linear combination
  - Formula: `hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic`
  - `alpha` parameter controls weighting: 0.0 = pure semantic search, 1.0 = pure BM25, 0.5 = equal weighting (default)
  - Both BM25 and semantic scores are normalized to [0, 1] range before combination
  - Searches top 2500 results from each method, normalizes scores, combines, and returns top `--limit` results
  - Output format: prints movie titles only
- **Reciprocal Rank Fusion (RRF)**: `rrf-search` command combines BM25 and semantic search results using rank-based fusion
  - Formula: `rrf_score = 1 / (k + rank)` where `rank` is the position (1-indexed) in each result list
  - `k` parameter controls the fusion constant (default 60); higher k values reduce the impact of rank differences
  - Documents appearing in both result sets have their RRF scores summed
  - Uses ranks (position) rather than normalized scores, making it robust to different score distributions
  - Searches top `limit * 500` results from each method, calculates RRF scores, and returns top `--limit` results sorted by RRF score
  - Output format: prints movie title, RRF score, and description preview
- **Score Normalization**: `normalize` command performs min-max normalization to scale scores to [0, 1] range using formula `(score - min) / (max - min)`
  - If all scores are identical, returns all 1.0 values
  - If no scores provided, prints nothing
  - Example: `normalize 0.5 2.3 1.2 0.5 0.1` outputs normalized scores with 4 decimal places

### GPU/CUDA
- The CLI runs on CPU by default and avoids initializing CUDA to prevent GPU capability mismatches.
