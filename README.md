# Hoopla

A movie search engine that uses advanced text search algorithms to find relevant movies from a large dataset.

## What is Hoopla?

Hoopla is a command-line tool that searches through movie descriptions using sophisticated text ranking algorithms. It helps you find movies by analyzing how well search terms match movie titles and descriptions, then ranks results by relevance.

## Why Hoopla?

- **Smart Search**: Uses BM25 algorithm (the same technology behind Google search) to find the most relevant movies
- **Fast Performance**: Pre-built search index means instant results
- **Educational**: Demonstrates how modern search engines work under the hood
- **Flexible**: Supports different search scoring methods (TF-IDF, BM25) for learning and experimentation

## How it Works

### Core Components

1. **Text Processing**: Converts movie titles and descriptions into searchable tokens
   - Removes punctuation and converts to lowercase
   - Filters out common words (stopwords) like "the", "and", "is"
   - Stems words to their root form (e.g., "running" → "run")

2. **Search Index**: Builds an inverted index that maps words to movies containing them
   - Stores which movies contain each word
   - Tracks how often words appear in each movie
   - Calculates document lengths for ranking

3. **Ranking Algorithms**: 
   - **TF-IDF**: Basic relevance scoring
   - **BM25**: Advanced algorithm that considers term frequency, document length, and collection statistics

### Data Structure

The system works with movie data containing:
- Movie ID
- Title
- Description (plot summary)

## Getting Started

### Prerequisites

- Python 3.13+
- NLTK library for text processing

### Installation

```bash
# Install dependencies
uv sync

# Build the search index (required before searching)
python cli/keyword_search_cli.py build
```

### Usage

#### Search Movies
```bash
# Search using BM25 algorithm (recommended)
python cli/keyword_search_cli.py bm25search "action thriller"

# Search with custom result limit
python cli/keyword_search_cli.py bm25search "comedy romance" --limit 10
```

#### Explore Search Components
```bash
# Get term frequency for a word in a specific movie
python cli/keyword_search_cli.py tf 1 "action"

# Get inverse document frequency for a word
python cli/keyword_search_cli.py idf "thriller"

# Get TF-IDF score
python cli/keyword_search_cli.py tfidf 1 "action"

# Get BM25 components
python cli/keyword_search_cli.py bm25idf "thriller"
python cli/keyword_search_cli.py bm25tf 1 "action"
```

## Project Structure

```
hoopla/
├── cli/                    # Command-line interface
│   ├── keyword_search_cli.py    # Main CLI script
│   └── lib/                     # Core search functionality
│       ├── keyword_search.py    # Search algorithms and index
│       └── search_utils.py      # Utilities and data loading
├── data/                   # Movie dataset and stopwords
│   ├── movies.json         # Movie database
│   └── stopwords.txt       # Common words to filter out
├── cache/                  # Pre-built search index (auto-generated)
└── pyproject.toml          # Project configuration
```

## Technical Details

### BM25 Algorithm Parameters
- **k1**: Controls term frequency saturation (default: 1.5)
- **b**: Controls length normalization (default: 0.75)

### Performance Features
- **Caching**: Search index is built once and cached for fast subsequent searches
- **Efficient Storage**: Uses pickle files for fast loading of pre-computed data
- **Memory Optimization**: Only loads necessary components when needed

## Troubleshooting

**Index not found error**: Run `python cli/keyword_search_cli.py build` first

**No search results**: Try broader search terms or check if the movie database is properly loaded

**Memory issues**: The system loads the entire index into memory for fast searching
