import string

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    with open("data/stopwords.txt", "r") as f:
        stopwords = f.read().splitlines()
    for movie in movies:
        query_tokens = remove_stopwords(tokenize_text(query), stopwords)
        title_tokens = remove_stopwords(tokenize_text(movie["title"]), stopwords)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def remove_stopwords(tokens, stopwords):
        return [token for token in tokens if token not in stopwords]


def preprocess_text(text: str) -> str: 
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]: 
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens: 
        if token: 
            valid_tokens.append(token)
    return valid_tokens

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens: 
        for title_token in title_tokens: 
            if query_token in title_token: 
                return True
    return False




