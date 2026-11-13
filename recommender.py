"""Loads the saved artifacts and serves recommendations."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Recommender:
    def __init__(self, art_dir="artifacts"):
        art = Path(art_dir)
        self.movies = pd.read_pickle(art / "movies.pkl")
        self.X = sp.load_npz(art / "vectors.npz").tocsr()
        self.labels = self._make_labels()
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}

    def _make_labels(self):
        """Unique 'Title (year)' labels for the dropdown."""
        labels, seen = [], set()
        for r in self.movies.itertuples():
            year = f" ({int(r.year)})" if pd.notna(r.year) else ""
            label = f"{r.title}{year}"
            if label in seen:
                label = f"{label} #{r.id}"
            seen.add(label)
            labels.append(label)
        return labels

    def info(self, idx):
        m = self.movies.iloc[idx]
        return {
            "id": int(m["id"]),
            "title": m["title"],
            "year": int(m["year"]) if pd.notna(m["year"]) else None,
            "rating": float(m["vote_average"]),
            "votes": int(m["vote_count"]),
            "genres": list(m["genres"]),
            "director": m["director"],
            "overview": m["overview"],
        }

    @staticmethod
    def explain(q, m):
        """Human-readable reasons a movie matched: the features the two share."""
        reasons = []
        genres = [g for g in m["genres"] if g in set(q["genres"])]
        if genres:
            reasons.append("Genres: " + ", ".join(genres))
        if m["director"] and m["director"] == q["director"]:
            reasons.append("Director: " + m["director"])
        cast = [c for c in m["cast"] if c in set(q["cast"])]
        if cast:
            reasons.append("Cast: " + ", ".join(cast))
        kws = [k for k in m["keywords"] if k in set(q["keywords"])][:3]
        if kws:
            reasons.append("Themes: " + ", ".join(kws))
        return reasons or ["Similar plot description"]

    def recommend(self, idx, n=10):
        sims = (self.X @ self.X[idx].T).toarray().ravel()
        sims[idx] = -1.0  # never recommend the movie itself
        n = min(n, len(sims) - 1)
        top = np.argpartition(-sims, n)[:n]
        top = top[np.argsort(-sims[top])]
        q = self.movies.iloc[idx]
        results = []
        for j in top:
            info = self.info(int(j))
            info["score"] = float(sims[j])
            info["why"] = self.explain(q, self.movies.iloc[int(j)])
            results.append(info)
        return results
