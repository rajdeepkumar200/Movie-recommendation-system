"""CLI: print top-N recommendations for one or more users.

Loads the cached hybrid model if it exists, otherwise trains a fresh one.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .collaborative import MatrixFactorization, MFConfig
from .data import (
    build_item_features,
    build_user_features,
    load_ml100k,
    train_test_split_random,
)
from .hybrid import HybridRecommender
from .supervised import SupervisedConfig, SupervisedRecommender
from .svdpp import SVDpp, SVDppConfig

CACHE_PATH = Path("data") / "hybrid_model.pkl"


def parse_args():
    p = argparse.ArgumentParser(description="Recommend top-N movies for a user.")
    p.add_argument("--user", type=int, nargs="*", default=None,
                   help="User id(s) (0-indexed). Default: 3 random users.")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--cf-model", choices=["mf", "svdpp"], default="svdpp")
    p.add_argument("--factors", type=int, default=80)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrain", action="store_true", help="Force retrain, ignore cache.")
    return p.parse_args()


def train_hybrid(args, ds, train):
    if args.cf_model == "svdpp":
        cf = SVDpp(
            ds.n_users, ds.n_items,
            SVDppConfig(n_factors=args.factors, n_epochs=args.epochs,
                        lr=args.lr, reg=args.reg, seed=args.seed),
        )
    else:
        cf = MatrixFactorization(
            ds.n_users, ds.n_items,
            MFConfig(n_factors=args.factors, n_epochs=args.epochs,
                     lr=args.lr, reg=args.reg, seed=args.seed),
        )
    t0 = time.time()
    cf.fit(train["user"].to_numpy(), train["item"].to_numpy(), train["rating"].to_numpy())
    print(f"CF ({args.cf_model}) trained in {time.time() - t0:.1f}s")

    user_feats = build_user_features(ds.users, ds.n_users)
    item_feats = build_item_features(ds.items, ds.n_items)
    sup = SupervisedRecommender(user_feats, item_feats, SupervisedConfig(seed=args.seed))
    cf_pred = cf.predict(train["user"].to_numpy(), train["item"].to_numpy())
    t0 = time.time()
    sup.fit(train["user"].to_numpy(), train["item"].to_numpy(),
            train["rating"].to_numpy(), cf_pred)
    print(f"Supervised trained in {time.time() - t0:.1f}s")

    return HybridRecommender(cf, sup, alpha=args.alpha)


def show_recommendations(hybrid, ds, train, user, top):
    item_titles = dict(zip(ds.items["item"].astype(int), ds.items["title"]))
    seen_items = set(train.loc[train["user"] == user, "item"].astype(int).tolist())

    # Score every item, mask out the ones the user already rated, take top-N.
    all_items = np.arange(ds.n_items, dtype=np.int64)
    users_arr = np.full_like(all_items, user)
    scores = hybrid.predict(users_arr, all_items)
    scores[list(seen_items)] = -np.inf
    top_idx = np.argpartition(-scores, kth=top - 1)[:top]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    print(f"\n=== Top-{top} recommendations for user {user} ===")
    print(f"  ({len(seen_items)} items already rated in training)")
    for rank, i in enumerate(top_idx, 1):
        title = item_titles.get(int(i), f"item#{i}")
        print(f"  {rank:2d}. [pred {scores[i]:.2f}]  {title}")

    # A bit of context - what has this user been rating recently?
    user_train = (
        train[train["user"] == user]
        .sort_values("timestamp", ascending=False)
        .head(5)
    )
    if not user_train.empty:
        print(f"\n  Recent ratings by user {user}:")
        for _, row in user_train.iterrows():
            title = item_titles.get(int(row["item"]), f"item#{int(row['item'])}")
            print(f"    {row['rating']:.0f}/5  {title}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Loading MovieLens 100K...")
    ds = load_ml100k("data")
    train, _test = train_test_split_random(ds.ratings, test_frac=0.2, seed=args.seed)

    hybrid = None
    if CACHE_PATH.exists() and not args.retrain:
        try:
            print(f"Loading cached model from {CACHE_PATH}...")
            with CACHE_PATH.open("rb") as f:
                hybrid = pickle.load(f)
        except Exception as e:
            print(f"  cache load failed ({e}); retraining.")
            hybrid = None

    if hybrid is None:
        hybrid = train_hybrid(args, ds, train)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("wb") as f:
            pickle.dump(hybrid, f)
        print(f"Cached model to {CACHE_PATH}")

    if args.user:
        users = args.user
    else:
        # Pick a handful of random users to demo on.
        rng = np.random.default_rng(args.seed)
        users = rng.choice(ds.n_users, size=3, replace=False).tolist()

    for u in users:
        show_recommendations(hybrid, ds, train, int(u), args.top)


if __name__ == "__main__":
    main()
