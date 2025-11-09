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

# Optional: For query enhancement and RAG, create a .env file with:
# GEMINI_API_KEY=your_api_key_here
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

# RRF search with query enhancement (spell correction)
python cli/hybrid_search_cli.py rrf-search "briish bear" --enhance spell

# RRF search with query enhancement (query rewriting)
python cli/hybrid_search_cli.py rrf-search "bear movie that gives me the lulz" --enhance rewrite

# RRF search with query enhancement (query expansion)
python cli/hybrid_search_cli.py rrf-search "scary bear movie" --enhance expand

# RRF search with reranking (uses LLM to score and rerank results)
python cli/hybrid_search_cli.py rrf-search "action movie" --rerank-method individual --limit 5

# RRF search with batch reranking (faster, single LLM call)
python cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method batch --limit 3

# RRF search with cross-encoder reranking (fastest, no API key required, runs locally)
python cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --rerank-method cross_encoder --limit 25

# RRF search with LLM evaluation (rates results 0-3 for relevance)
python cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --evaluate

# Normalize scores using min-max normalization
python cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1

# Evaluate search performance using golden dataset
python cli/evaluation_cli.py --limit 3
python cli/evaluation_cli.py --limit 6

# RAG (Retrieval Augmented Generation) - search + generate answer
python cli/augmented_generation_cli.py rag "movies about time travel"

# Summarize search results
python cli/augmented_generation_cli.py summarize "action movies" --limit 10

# Summarize with citations
python cli/augmented_generation_cli.py citations "movies about time travel" --limit 10
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
- `rrf-search <query> [--k <int>] [--limit <int>] [--enhance <method>] [--rerank-method <method>] [--evaluate]` - Reciprocal Rank Fusion (RRF) hybrid search with configurable k constant (default 60). Optional `--enhance spell` enables spell correction, `--enhance rewrite` enables query rewriting, `--enhance expand` expands queries with related terms using Gemini API. Optional `--rerank-method individual` uses LLM to score each result individually, `--rerank-method batch` uses LLM to rerank all results in a single call (faster, fewer API calls), or `--rerank-method cross_encoder` uses a local neural network model for reranking (fastest, no API key required). Optional `--evaluate` uses LLM to rate result relevance on a 0-3 scale (3=Highly relevant, 2=Relevant, 1=Marginally relevant, 0=Not relevant)
- `normalize <scores...>` - Normalize scores using min-max normalization to range [0, 1]

### Retrieval Augmented Generation (RAG) Commands
- `rag <query>` - Perform RAG: search for relevant movies using RRF search, then generate a comprehensive answer using Gemini API. Returns search results and AI-generated answer tailored for Hoopla users. Requires `GEMINI_API_KEY` in `.env` file or environment variables.
- `summarize <query> [--limit <int>]` - Summarize search results: search for relevant movies using RRF search, then generate a concise summary synthesizing information from multiple results. Returns search results and AI-generated summary (3-4 sentences) that combines information from multiple sources. Default limit is 5. Requires `GEMINI_API_KEY` in `.env` file or environment variables.
- `citations <query> [--limit <int>]` - Summarize search results with citations: search for relevant movies using RRF search, then generate a comprehensive answer with source citations using [1], [2], etc. format. Cites sources when referencing information, mentions different viewpoints if sources disagree, and indicates when insufficient information is available. Default limit is 5. Requires `GEMINI_API_KEY` in `.env` file or environment variables.

### Evaluation Commands
- `evaluation_cli.py [--limit <int>]` - Evaluate search performance using a golden dataset. Calculates precision@k, recall@k, and F1 score for each test query by comparing retrieved results against expected relevant documents. Default limit is 5. Requires `data/golden_dataset.json` with test cases containing queries and relevant document titles.

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
  - By default, searches top `limit` results from each method, calculates RRF scores, and returns top `--limit` results sorted by RRF score
  - Output format: prints movie title, RRF score, BM25 rank, semantic rank, and description preview
  - **Reranking**: `--rerank-method` option reranks results for improved relevance. Three methods available:
    - `individual`: Uses LLM to score each result individually on a 0-10 scale (slower, more API calls)
      - When enabled, gathers `limit * 5` results from RRF search (instead of just `limit`)
      - LLM scores each result individually on a 0-10 scale based on query relevance
      - Results are sorted by LLM rerank score (individual_score) in descending order and top `limit` are returned
      - Includes 3-second delay between LLM calls to respect rate limits
      - Output shows individual score (0-10) alongside RRF score for each result
      - Requires `GEMINI_API_KEY` in `.env` file or environment variables
    - `batch`: Uses LLM to rerank all results in a single call (faster, fewer API calls)
      - When enabled, gathers `limit * 5` results from RRF search (instead of just `limit`)
      - LLM receives all documents in a single prompt and returns a JSON list of IDs ranked by relevance
      - Results are sorted by LLM's ranking order and top `limit` are returned
      - Output shows rerank rank (1-indexed position from LLM ranking) alongside RRF score for each result
      - Requires `GEMINI_API_KEY` in `.env` file or environment variables
    - `cross_encoder`: Uses a local neural network model (cross-encoder/ms-marco-TinyBERT-L2-v2) for reranking (fastest, no API key required)
      - When enabled, gathers `limit * 5` results from RRF search (instead of just `limit`)
      - Cross-encoder model scores each query-document pair for relevance
      - Results are sorted by cross-encoder score in descending order and top `limit` are returned
      - Output shows cross encoder score alongside RRF score for each result
      - No API key required, runs entirely locally
  - **Query Enhancement**: `--enhance` option enables automatic query improvement using Google's Gemini API
    - `--enhance spell`: Corrects spelling errors in search queries (e.g., "briish bear" → "british bear")
    - `--enhance rewrite`: Rewrites vague queries to be more searchable and specific (e.g., "bear movie that gives me the lulz" → "Comedy bear movie Ted style")
    - `--enhance expand`: Expands queries with synonyms and related terms to improve search coverage (e.g., "scary bear movie" → "scary horror grizzly bear movie terrifying film")
    - Requires `GEMINI_API_KEY` in `.env` file or environment variables
    - Falls back to original query if enhancement fails or API key is not set
  - **Result Evaluation**: `--evaluate` option uses LLM to rate search result relevance after displaying results
    - Scores each result on a 0-3 scale: 3=Highly relevant, 2=Relevant, 1=Marginally relevant, 0=Not relevant
    - Prints evaluation report in format: `1. Movie Title: 2/3`
    - Runs after search results are displayed
    - Requires `GEMINI_API_KEY` in `.env` file or environment variables
    - Returns all 0 scores if API key is not set
- **Score Normalization**: `normalize` command performs min-max normalization to scale scores to [0, 1] range using formula `(score - min) / (max - min)`
  - If all scores are identical, returns all 1.0 values
  - If no scores provided, prints nothing
  - Example: `normalize 0.5 2.3 1.2 0.5 0.1` outputs normalized scores with 4 decimal places

### Evaluation
- **Golden Dataset**: Evaluation uses `data/golden_dataset.json` containing test cases with queries and expected relevant document titles
- **Precision@k**: Calculates precision at k by dividing the number of relevant documents retrieved in the top k results by k
- **Recall@k**: Calculates recall at k by dividing the number of relevant documents retrieved in the top k results by the total number of relevant documents
- **F1 Score**: Calculates the harmonic mean of precision and recall using the formula `2 * (precision * recall) / (precision + recall)`. Returns 0 if both precision and recall are 0
- **Output Format**: For each test query, displays the query, precision@k score, recall@k score, F1 score, retrieved document titles (top k), and expected relevant document titles
- **RRF Search**: Evaluation uses RRF search (k=60) to retrieve results for each test query

### Retrieval Augmented Generation (RAG)
- Uses RRF search (k=60) to retrieve relevant movies for the query
- Searches `limit * SEARCH_MULTIPLIER` results (default limit=5, multiplier=5, so 25 results)
- Generates context from search results by combining movie titles and descriptions
- Uses Gemini 2.0 Flash model to generate answers/summaries based on retrieved context
- Tailored for Hoopla users (movie streaming service context)
- **RAG command**: Generates comprehensive answers addressing the query
- **Summarize command**: Generates concise 3-4 sentence summaries synthesizing information from multiple sources, information-dense with key details about genre, plot, etc.
- **Citations command**: Generates comprehensive answers with source citations using [1], [2], etc. format. Cites sources when referencing information, mentions different viewpoints if sources disagree, and indicates when insufficient information is available
- Output format: displays search result titles, then AI-generated answer/summary (citations command shows numbered sources corresponding to search result order)
- Requires `GEMINI_API_KEY` in `.env` file or environment variables
- Returns error if no search results found

### GPU/CUDA
- The CLI runs on CPU by default and avoids initializing CUDA to prevent GPU capability mismatches.
