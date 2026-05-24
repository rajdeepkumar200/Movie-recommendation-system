"""MovieLens-100K loader + a couple of feature helpers."""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Order matters - this is the column order in u.item.
GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
    "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western",
]


@dataclass
class Dataset:
    ratings: pd.DataFrame   # user, item, rating, timestamp (ids re-indexed 0..N-1)
    users: pd.DataFrame
    items: pd.DataFrame
    n_users: int
    n_items: int


def download_ml100k(data_dir="data"):
    data_dir = Path(data_dir)
    target = data_dir / "ml-100k"
    if target.exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens 100K to {data_dir}...")
    r = requests.get(ML100K_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(data_dir)
    return target


def load_ml100k(data_dir="data") -> Dataset:
    root = download_ml100k(data_dir)

    ratings = pd.read_csv(
        root / "u.data", sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    users = pd.read_csv(
        root / "u.user", sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
    )
    # latin-1: a few titles have accents that aren't valid UTF-8
    item_cols = ["item_id", "title", "release_date", "video_release", "imdb_url"] + GENRE_COLS
    items = pd.read_csv(root / "u.item", sep="|", names=item_cols, encoding="latin-1")

    # The raw ids are 1-indexed and may have gaps - remap them to a dense range.
    u_map = {u: i for i, u in enumerate(sorted(ratings["user_id"].unique()))}
    i_map = {it: i for i, it in enumerate(sorted(ratings["item_id"].unique()))}

    ratings = ratings.assign(
        user=ratings["user_id"].map(u_map).astype(np.int32),
        item=ratings["item_id"].map(i_map).astype(np.int32),
        rating=ratings["rating"].astype(np.float32),
    )[["user", "item", "rating", "timestamp"]]

    users = users.assign(user=users["user_id"].map(u_map)).dropna(subset=["user"])
    users["user"] = users["user"].astype(np.int32)
    items = items.assign(item=items["item_id"].map(i_map)).dropna(subset=["item"])
    items["item"] = items["item"].astype(np.int32)

    return Dataset(
        ratings=ratings.reset_index(drop=True),
        users=users.reset_index(drop=True),
        items=items.reset_index(drop=True),
        n_users=len(u_map),
        n_items=len(i_map),
    )


def train_test_split_random(ratings, test_frac=0.2, seed=42):
    """Plain random split. This is what most MovieLens papers report against."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ratings))
    n_test = int(round(len(ratings) * test_frac))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return (
        ratings.iloc[train_idx].reset_index(drop=True),
        ratings.iloc[test_idx].reset_index(drop=True),
    )


def train_test_split_by_time(ratings, test_frac=0.2, seed=42):
    """Per-user time split: the user's most recent ratings go into test.

    Users with fewer than 5 ratings stay entirely in train - splitting them
    leaves the train side with nothing useful for that user.
    """
    train_idx, test_idx = [], []
    for _, grp in ratings.groupby("user", sort=False):
        n = len(grp)
        if n < 5:
            train_idx.extend(grp.index.tolist())
            continue
        order = grp.sort_values("timestamp").index.to_numpy()
        n_test = max(1, int(round(n * test_frac)))
        test_idx.extend(order[-n_test:].tolist())
        train_idx.extend(order[:-n_test].tolist())
    train = ratings.loc[train_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = ratings.loc[test_idx].reset_index(drop=True)
    return train, test


def build_item_features(items, n_items):
    # (n_items, n_genres) binary matrix, indexed by item id.
    feats = np.zeros((n_items, len(GENRE_COLS)), dtype=np.float32)
    for _, row in items.iterrows():
        feats[int(row["item"])] = row[GENRE_COLS].to_numpy(dtype=np.float32)
    return feats


def build_user_features(users, n_users):
    # age (normalized to ~[0,1]) + gender bit + occupation one-hot.
    occupations = sorted(users["occupation"].unique())
    occ_idx = {o: i for i, o in enumerate(occupations)}
    d = 2 + len(occupations)
    feats = np.zeros((n_users, d), dtype=np.float32)
    for _, row in users.iterrows():
        u = int(row["user"])
        feats[u, 0] = float(row["age"]) / 100.0
        feats[u, 1] = 1.0 if row["gender"] == "M" else 0.0
        feats[u, 2 + occ_idx[row["occupation"]]] = 1.0
    return feats
