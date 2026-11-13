"""Evaluate the recommender with a fair, non-circular test.

Problem: genres are part of the final model's features, so measuring genre overlap on
that model would be circular. Instead we compare feature variants that EXCLUDE genres,
then ask: do the recommendations still share genres with the query movie?
  - share@10 : fraction of top-10 recs that share at least one genre with the query
  - jaccard@10 : mean genre-set Jaccard between the query and its top-10 recs
A random-recommendation baseline shows what "no skill" looks like.

Also reports catalog coverage and latency for the final model.

Run:  python evaluate.py        (needs data/ CSVs, like build_model.py)
"""
import time

import numpy as np
import pandas as pd

from build_model import build_docs, load_movies, vectorize

K = 10
N_QUERIES = 500
SEED = 42


def top_k(X, queries, k=K):
    sims = (X[queries] @ X.T).toarray()
    sims[np.arange(len(queries)), queries] = -1.0
    return np.argpartition(-sims, k, axis=1)[:, :k]


def genre_scores(genres, queries, recs):
    share, jac = [], []
    for q, row in zip(queries, recs):
        gq = genres[q]
        for j in row:
            gj = genres[j]
            union = gq | gj
            share.append(bool(gq & gj))
            jac.append(len(gq & gj) / len(union) if union else 0.0)
    return float(np.mean(share)), float(np.mean(jac))


def main():
    df = load_movies()
    genres = [set(g) for g in df.genres]
    rng = np.random.default_rng(SEED)
    eligible = np.array([i for i, g in enumerate(genres) if g])
    queries = rng.choice(eligible, size=min(N_QUERIES, len(eligible)), replace=False)

    rows = []

    # Random baseline
    rand = np.array([rng.choice(len(df), size=K, replace=False) for _ in queries])
    s, j = genre_scores(genres, queries, rand)
    rows.append(("Random baseline", s, j))

    no_genre = dict(genres=False)
    variants = [
        ("Overview only, TF-IDF", dict(genres=False, keywords=False, cast=False, director=False), "tfidf"),
        ("Overview+keywords+cast+director, Count", no_genre, "count"),
        ("Overview+keywords+cast+director, TF-IDF", no_genre, "tfidf"),
    ]
    for name, flags, kind in variants:
        X, _ = vectorize(build_docs(df, **flags), kind)
        s, j = genre_scores(genres, queries, top_k(X, queries))
        rows.append((name, s, j))

    result = pd.DataFrame(rows, columns=["variant (genres excluded from features)", "share@10", "jaccard@10"])
    print(f"\nGenre agreement of top-{K} recommendations, {len(queries)} random query movies\n")
    print(result.round(3).to_string(index=False))
    result.to_csv("eval_results.csv", index=False)

    # Final model (includes genres): coverage + latency
    X, _ = vectorize(build_docs(df), "tfidf")
    seen = set()
    for start in range(0, len(df), 500):
        chunk = np.arange(start, min(start + 500, len(df)))
        seen.update(top_k(X, chunk).ravel().tolist())
    t0 = time.time()
    for q in queries[:100]:
        sims = (X @ X[q].T).toarray().ravel()
        np.argpartition(-sims, K)[:K]
    ms = (time.time() - t0) / 100 * 1000
    print(f"\nFinal model catalog coverage@{K}: {len(seen) / len(df):.1%} of movies appear in at least one top-{K} list")
    print(f"Average query time: {ms:.1f} ms")
    print("\nSaved eval_results.csv")


if __name__ == "__main__":
    main()
