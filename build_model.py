"""Build the recommender from the TMDB 5000 CSVs.

Expects:  data/tmdb_5000_movies.csv  and  data/tmdb_5000_credits.csv
Run:      python build_model.py
Creates:  artifacts/movies.pkl (movie metadata) and artifacts/vectors.npz (L2-normalized
          sparse feature matrix, so cosine similarity is just a dot product).
"""
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.preprocessing import normalize

DATA_DIR = Path("data")
ART_DIR = Path("artifacts")
_stemmer = PorterStemmer()


# ---------- parsing the TMDB JSON-like columns ----------
def parse_names(value, limit=None):
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []
    names = [d["name"] for d in items if "name" in d]
    return names[:limit] if limit else names


def parse_director(value):
    try:
        crew = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return ""
    for member in crew:
        if member.get("job") == "Director":
            return member.get("name", "")
    return ""


def load_movies(data_dir=DATA_DIR):
    """Merge movies + credits into one clean DataFrame."""
    movies = pd.read_csv(Path(data_dir) / "tmdb_5000_movies.csv")
    credits = (
        pd.read_csv(Path(data_dir) / "tmdb_5000_credits.csv")
        .rename(columns={"movie_id": "id"})
        .drop(columns=["title"])
    )
    df = movies.merge(credits, on="id")
    year = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    out = pd.DataFrame(
        {
            "id": df["id"],
            "title": df["title"],
            "year": year.astype("Int64"),
            "overview": df["overview"].fillna(""),
            "genres": df["genres"].apply(parse_names),
            "keywords": df["keywords"].apply(parse_names),
            "cast": df["cast"].apply(lambda s: parse_names(s, 3)),  # top-billed 3
            "director": df["crew"].apply(parse_director),
            "vote_average": df["vote_average"],
            "vote_count": df["vote_count"],
        }
    )
    return out.drop_duplicates(subset="id").reset_index(drop=True)


# ---------- turning each movie into one text "document" ----------
def squash(name):
    """'Science Fiction' -> 'sciencefiction', so multi-word names stay one token."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def stem_overview(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [_stemmer.stem(w) for w in words if w not in ENGLISH_STOP_WORDS]


def build_docs(df, overview=True, genres=True, keywords=True, cast=True, director=True):
    docs = []
    for r in df.itertuples():
        parts = []
        if overview:
            parts += stem_overview(r.overview)
        if genres:
            parts += [squash(g) for g in r.genres]
        if keywords:
            parts += [squash(k) for k in r.keywords]
        if cast:
            parts += [squash(c) for c in r.cast]
        if director and r.director:
            parts.append(squash(r.director))
        docs.append(" ".join(p for p in parts if p))
    return docs


def vectorize(docs, kind="tfidf"):
    """Return an L2-normalized sparse matrix (rows = movies) and the fitted vectorizer."""
    if kind == "count":
        vec = CountVectorizer(min_df=2)
    else:
        vec = TfidfVectorizer(min_df=2, sublinear_tf=True)
    X = vec.fit_transform(docs)
    return normalize(X).astype(np.float32).tocsr(), vec


def main():
    ART_DIR.mkdir(exist_ok=True)
    df = load_movies()
    print(f"Loaded {len(df)} movies")

    X, vec = vectorize(build_docs(df), kind="tfidf")
    df.to_pickle(ART_DIR / "movies.pkl")
    sp.save_npz(ART_DIR / "vectors.npz", X)

    print(f"Feature matrix: {X.shape[0]} movies x {X.shape[1]} features, {X.nnz} non-zeros")
    print(f"Saved artifacts to ./{ART_DIR}/ (small enough to commit to GitHub)")


if __name__ == "__main__":
    main()
