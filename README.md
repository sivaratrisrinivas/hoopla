# Bingo

Bingo is a terminal program that finds movies from a local plot list by matching the words you type and by matching the meaning of what you meant.

## Who it is for

Use Bingo if you remember a plot and not the title. Ordinary word search looks for the same words you typed. That fails when you search `cute british bear marmalade` and the title is Paddington. Bingo can still find that film because it also compares meaning.

Use it if you are measuring those two styles of search against each other. The repo already has a 10-query check on 5000 movies, so you can see which method ranked the known-good titles higher. It runs on your machine. There is no hosted demo.

## How to try it

You need Python 3.13 or newer. [uv](https://docs.astral.sh/uv/) is the installer this repo is set up for.

```bash
git clone https://github.com/sivaratrisrinivas/Bingo
cd Bingo
uv sync
```

Get the movie list and the stopword file. Stopwords are tiny words such as "the" and "and" that search skips.

```bash
mkdir -p data
curl -L -o data/movies.json "https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/movies.json"
cp eval/fixtures/stopwords.txt data/stopwords.txt
```

Build the word-search lookup, then search. That first build took 16.31 seconds in the measured run.

```bash
uv run python cli/keyword_search_cli.py build
uv run python cli/keyword_search_cli.py bm25search "cute british bear marmalade" --limit 5
```

To search by meaning, precompute a number list for every movie description. That step took 337.05 seconds in the same run.

```bash
uv run python cli/semantic_search_cli.py embed_chunks
uv run python cli/semantic_search_cli.py search_chunked "cute british bear marmalade" --limit 5
```

To merge the word list and the meaning list:

```bash
uv run python cli/hybrid_search_cli.py rrf-search "cute british bear marmalade" --limit 5
```

Spell correction, query rewriting, and AI-written answers need `GEMINI_API_KEY` in a `.env` file. Word search and meaning search do not.

## What the numbers mean

On 2026-08-24 the repo scored `data/golden_dataset.json`. That run used 10 queries, 5000 movies, and k=5. k is how many top results we scored. The machine was an Intel Xeon with 4 CPUs and 15.64 GB of RAM, running Python 3.13.15. Meaning search used `all-MiniLM-L6-v2`. The second-pass ranker was `cross-encoder/ms-marco-TinyBERT-L2-v2`.

Three quality scores are averaged across those 10 queries.

**Precision@k.** Of the k movies the search returned, what share were on the known-good list. Precision 0.3800 at k=5 means about 1.9 of those 5 hits were right.

**Recall@k.** Of every movie on the known-good list, what share showed up in those k results. Recall 0.5157 means the search found about half of the titles it was supposed to find. The rest ranked worse than 5th, or were missing from the movie file.

**F1.** One number that balances precision and recall. If the search is clean but incomplete, or noisy but complete, F1 falls.

Speed is **p95 query latency** in milliseconds. For this run, 95 percent of searches finished in that time or faster.

| Configuration | Precision@5 | Recall@5 | F1 | p95 query latency (ms) |
|---|---|---|---|---|
| BM25 | 0.3800 | 0.5157 | 0.3607 | 3419.2 |
| Semantic | 0.3200 | 0.4606 | 0.3088 | 335.1 |
| RRF | 0.3800 | 0.5049 | 0.3599 | 3804.8 |
| RRF + cross-encoder | 0.3600 | 0.4824 | 0.3357 | 4038.2 |

BM25 is the word-matching scorer. It weighs how often your words appear, how rare they are, and how long the plot is. Semantic is meaning search. It turns text into a list of numbers and finds plots whose lists are close to the query's list. RRF, Reciprocal Rank Fusion, merges the two ranked lists by position rather than by raw score. RRF + cross-encoder takes that merged list and re-scores each movie against the query with a second model.

On this 10-query set, word matching won. Meaning search was much faster, 335.1 ms against 3419.2 ms, and a bit worse on every quality score. Merging the lists tied word matching on precision and did not beat it on recall or F1. The second-pass ranker made both quality and speed worse.

Ten queries is a small sample. Read the table as a measurement of this run, not a ranking of methods for every search.

Index build time: 353.36s (BM25 16.31s, chunk embeddings 337.05s, cold rebuild). The eval always rebuilds the HybridSearch index it scores, so a cache hit cannot rewrite this number to ~0.

The known-good list marks `Død snø` as relevant for the zombie query. That title is not in `movies.json`, so no method can retrieve it.

Reproduce from a clean checkout. This downloads `data/movies.json` if it is missing, copies the known-good list and stopwords from `eval/fixtures`, then writes `eval/results.json` and prints the table:

```bash
uv run python eval/run_retrieval_metrics.py
```

## Install notes

The steps above use uv. pip works too:

```bash
pip install -e .
```

The product name is Bingo. The installable Python package name in `pyproject.toml` is still `hoopla`. Leave that package name as-is.

## Data setup

Movie data lives at `data/movies.json`:

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

`data/stopwords.txt` is one stopword per line. Copy it from `eval/fixtures/stopwords.txt` if you are not using the eval script. The eval script also copies `eval/fixtures/golden_dataset.json` to `data/golden_dataset.json`. If those files already exist and differ from the fixtures, the eval refuses to run.

## How it works

### Word search, BM25

1. Build an inverted index, a lookup from each word to the movies that contain it, like the index at the back of a book.
2. Clean and stem words. Stemming cuts "running" down to "run".
3. Score hits with BM25.

### Meaning search

1. Load the sentence model `all-MiniLM-L6-v2`.
2. Turn queries and plots into number lists, often called embeddings.
3. Rank movies whose number lists are close to the query. Closeness here is cosine similarity.
4. Split long plots into chunks so a matching sentence is not drowned by the rest of the text.

### Combined search

1. Run word search and meaning search on the same query.
2. Merge the two ranked lists.
3. Return the top of the merged list. Weighted merge mixes the scores. RRF mixes the ranks.

## Search commands

After `uv sync`, prefix these with `uv run python` or run them from the virtualenv `uv` created. Word search needs `build` first. Combined search and RAG need `embed_chunks` as well.

```bash
# Word search
python cli/keyword_search_cli.py bm25search "romantic comedy" --limit 10

# Meaning search
python cli/semantic_search_cli.py search "space adventure" --limit 5

# Combined search
python cli/hybrid_search_cli.py rrf-search "superhero action movie" --limit 10
```

```bash
# Spell correction
python cli/hybrid_search_cli.py rrf-search "briish bear" --enhance spell

# Rewrite a vague query into something searchable
python cli/hybrid_search_cli.py rrf-search "bear movie that gives me the lulz" --enhance rewrite

# Re-score the top hits with a local second-pass model
python cli/hybrid_search_cli.py rrf-search "action movie" --rerank-method cross_encoder --limit 25

# Rewrite a text query using an image you provide
python cli/describe_image_cli.py --image /path/to/your/image.jpg --query "find movies like this"

# Check that an image can be turned into a number list
python cli/multimodal_search_cli.py verify_image_embedding /path/to/your/image.jpg

# Search movies by image
python cli/multimodal_search_cli.py image_search /path/to/your/image.jpg
```

The code defaults image search to `data/paddington.jpeg`. That file is not in git. Pass a real image path.

### RAG, search then write an answer

RAG, retrieval-augmented generation, searches first, then asks Gemini to write from those hits. These commands need `GEMINI_API_KEY`.

```bash
python cli/augmented_generation_cli.py rag "movies about time travel"
python cli/augmented_generation_cli.py summarize "action movies" --limit 10
python cli/augmented_generation_cli.py citations "time travel movies" --limit 10
python cli/augmented_generation_cli.py question "What are some good horror movies?" --limit 10
```

## Command reference

### Word search (`keyword_search_cli.py`)

| Command | Description |
|---------|-------------|
| `build` | Create the word-search lookup from movie data |
| `search <query>` | Basic keyword search, first matches, not BM25-ranked |
| `bm25search <query>` | BM25-ranked search |
| `tf <doc_id> <term>` | Term frequency |
| `idf <term>` | Inverse document frequency |
| `tfidf <doc_id> <term>` | TF-IDF score |
| `bm25idf <term>` | BM25-IDF score |
| `bm25tf <doc_id> <term>` | BM25 term frequency |

### Meaning search (`semantic_search_cli.py`)

| Command | Description |
|---------|-------------|
| `verify` | Check that the sentence model loaded |
| `verify_embeddings` | Check movie embeddings and show stats |
| `search <query>` | Meaning search over whole plots |
| `embed_chunks` | Precompute chunk embeddings |
| `search_chunked <query> [--limit <int>]` | Meaning search over chunks |
| `chunk <text> [--chunk-size <int>] [--overlap <int>]` | Split into word chunks |
| `semantic_chunk <text> [--max-chunk-size <int>] [--overlap <int>]` | Split into sentence chunks |

### Combined search (`hybrid_search_cli.py`)

| Command | Description |
|---------|-------------|
| `weighted-search <query> [--alpha <float>] [--limit <int>]` | Mix scores. alpha 0.0 is all meaning, 1.0 is all BM25, default 0.5 |
| `rrf-search <query> [--k <int>] [--limit <int>] [--enhance <method>] [--rerank-method <method>] [--evaluate]` | Reciprocal Rank Fusion |
| `normalize <scores...>` | Min-max normalization to [0, 1] |

**RRF search options**

- `--enhance`: `spell` corrects typos, `rewrite` makes a vague query searchable, `expand` adds related terms
- `--rerank-method`: `individual` scores each hit with an LLM, `batch` reranks in one LLM call, `cross_encoder` uses a local model and needs no API key
- `--evaluate`: rate results 0-3 for relevance

The RRF `--k` flag is not the same k as precision@k. RRF k defaults to 60 and only changes how ranks are blended.

### RAG (`augmented_generation_cli.py`)

| Command | Description |
|---------|-------------|
| `rag <query>` | Search, then write a full answer |
| `summarize <query> [--limit <int>]` | 3-4 sentence summary |
| `citations <query> [--limit <int>]` | Answer with [1], [2] citations |
| `question <question> [--limit <int>]` | Casual Q&A |

Needs `GEMINI_API_KEY` in `.env` or the environment.

### Evaluation (`evaluation_cli.py`)

| Command | Description |
|---------|-------------|
| `python cli/evaluation_cli.py [--limit <int>]` | Score search on the known-good list, precision@k, recall@k, F1 |

Prefer `uv run python eval/run_retrieval_metrics.py` from a clean checkout. That is the path that copies fixtures, downloads movies if needed, and writes `eval/results.json`.

### Image query rewrite (`describe_image_cli.py`)

| Command | Description |
|---------|-------------|
| `--image <path>` | Image file, required |
| `--query <text>` | Text query to rewrite from the image, required |

Needs `GEMINI_API_KEY`.

### Image search (`multimodal_search_cli.py`)

| Command | Description |
|---------|-------------|
| `verify_image_embedding <image>` | Check image embedding generation, 512-dimensional CLIP embeddings |
| `image_search <image>` | Search movies by image, top 5 matches |

## Technical details

### Word search

- Library: NLTK for text processing and stemming
- Algorithm: BM25 with parameters k1=1.5, b=0.75
- Caching: processed data cached for later searches
- Features: custom search limits, detailed TF-IDF/BM25 scoring

### Meaning search

- Model: `all-MiniLM-L6-v2` from sentence-transformers
- Embeddings: 384-dimensional vectors
- Cache: `cache/movie_embeddings.npy`, rebuilt if the dataset size changes
- Context: titles plus descriptions
- Defaults: CPU device, search limit 5
- Chunking: default size 200 words, overlap 1. Sentence chunks default to 4 sentences

Precompute chunked embeddings:

```bash
python cli/semantic_search_cli.py embed_chunks
# Expected: "Generated 72909 chunked embeddings"
```

Rebuild cache:

```bash
rm cache/chunk_embeddings.npy cache/chunk_metadata.json
python cli/semantic_search_cli.py embed_chunks
```

### Combined search

**Weighted merge**

- Formula: `hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic`
- Searches top 2500 from each method, normalizes scores, combines, returns top `--limit`

**Reciprocal Rank Fusion**

- Formula: `rrf_score = 1 / (k + rank)` with default k=60
- Uses ranks rather than scores, so the two methods do not have to share a score scale
- A movie that appears in both lists gets the two RRF scores added

**Reranking**

- `individual`: an LLM scores each result on a 0-10 scale. Slower, more API calls
- `batch`: an LLM reranks all results in one call
- `cross_encoder`: a local neural net. Fastest of these, no API key

**Query enhancement**

- `spell`: corrects spelling errors
- `rewrite`: rewrites vague queries
- `expand`: adds synonyms and related terms

**Result evaluation**

- Rates results 0-3. 3 is highly relevant, 2 relevant, 1 marginally relevant, 0 not relevant

### RAG

- Search: RRF with k=60, retrieves `limit * 5` results, default 25
- Model: Gemini 2.0 Flash
- Context: movie titles plus descriptions
- Commands: `rag` writes a full answer, `summarize` writes 3-4 sentences, `citations` adds [1], [2] sources, `question` is casual Q&A
- Output: search result titles, then the generated text

### Evaluation

- Dataset: `data/golden_dataset.json`, queries plus expected titles
- Metrics: precision@k, recall@k, F1
- Method: RRF search with k=60 for the older single-method path. `eval/run_retrieval_metrics.py` scores BM25, semantic, RRF, and RRF plus cross-encoder

### Image query rewrite

- Model: Gemini 2.0 Flash
- Input: an image file plus a text query
- Process: reads the image, mixes that with the text query, rewrites a movie-search query
- Output: rewritten query plus token usage
- MIME types: auto-detects image format, JPEG, PNG, and similar, defaults to JPEG

### Image search

- Model: CLIP ViT-B-32 from sentence-transformers
- Embeddings: 512-dimensional vectors
- Device: CPU by default. CUDA is disabled to avoid GPU mismatches
- Search: compares the image embedding with movie title+description text embeddings using cosine similarity
- Format: movies encoded as "title: description" for CLIP text encoding

### Performance

- Device: CPU by default
- Caching: indexes and embeddings cached for later searches
- Defaults: search limit 5, RRF k=60, search multiplier 5

## Output format

**Chunked meaning search:**
```
1. The Incredibles (score: 0.8123)
   A family of undercover superheroes, while trying to live the quiet suburban life...
```

**RRF search:**
```
Movie Title (RRF score: 0.0234, BM25 rank: 3, Semantic rank: 5)
Description preview...
```

**RAG:**
```
Search Results:
  - Movie Title 1
  - Movie Title 2

RAG Response:
[AI-generated answer]
```

**Image query rewrite:**
```
Rewritten query: Paddington Bear family-friendly animated adventure movies
Total tokens:    1234
```

**Image search:**
```
Image search results for: data/paddington.jpeg
============================================================
1. Paddington (similarity: 0.309)
   A young Peruvian bear travels to London in search of a home...
2. Paddington 2 (similarity: 0.285)
   Paddington, now happily settled with the Brown family...
```

## Notes

- All search methods need the word-search lookup from the `build` command
- Combined search and RAG also need chunked embeddings from the `embed_chunks` command
- RAG and image query rewriting need `GEMINI_API_KEY`
- Cache files are rebuilt if the dataset size changes
- Search runs on CPU by default so a mismatched GPU cannot crash it
