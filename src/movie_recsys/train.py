"""Train the hybrid recommender end-to-end and print the metrics."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .collaborative import MatrixFactorization, MFConfig
from .data import (
    build_item_features,
    build_user_features,
    load_ml100k,
    train_test_split_by_time,
    train_test_split_random,
)
from .evaluate import ranking_eval_sampled, ranking_eval_vectorized
from .hybrid import HybridRecommender
from .metrics import mae, relevance_accuracy, rmse
from .supervised import SupervisedConfig, SupervisedRecommender
from .svdpp import SVDpp, SVDppConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train + evaluate the hybrid recommender.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--factors", type=int, default=80)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Weight on the CF prediction in the blend.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--threshold", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cf-model", choices=["mf", "svdpp"], default="svdpp")
    p.add_argument("--n-negatives", type=int, default=99)
    p.add_argument("--rank-threshold", type=float, default=4.0,
                   help="Min rating to count as positive for sampled-negatives P@K.")
    p.add_argument("--split", choices=["random", "time"], default="random")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Loading MovieLens 100K...")
    ds = load_ml100k(args.data_dir)
    print(f"  users={ds.n_users}  items={ds.n_items}  ratings={len(ds.ratings):,}")

    if args.split == "time":
        train, test = train_test_split_by_time(ds.ratings, test_frac=0.2, seed=args.seed)
    else:
        train, test = train_test_split_random(ds.ratings, test_frac=0.2, seed=args.seed)
    print(f"  split={args.split}  train={len(train):,}  test={len(test):,}")

    # ---- collaborative model
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
                     lr=args.lr, reg=args.reg, batch_size=args.batch_size, seed=args.seed),
        )
    t0 = time.time()
    cf.fit(train["user"].to_numpy(), train["item"].to_numpy(), train["rating"].to_numpy())
    print(f"CF ({args.cf_model}) trained in {time.time() - t0:.1f}s")

    # ---- supervised stacker
    user_feats = build_user_features(ds.users, ds.n_users)
    item_feats = build_item_features(ds.items, ds.n_items)
    sup = SupervisedRecommender(user_feats, item_feats, SupervisedConfig(seed=args.seed))
    cf_train_pred = cf.predict(train["user"].to_numpy(), train["item"].to_numpy())
    t0 = time.time()
    sup.fit(train["user"].to_numpy(), train["item"].to_numpy(),
            train["rating"].to_numpy(), cf_train_pred)
    print(f"Supervised trained in {time.time() - t0:.1f}s")

    hybrid = HybridRecommender(cf, sup, alpha=args.alpha)

    # ---- rating-prediction metrics
    y_true = test["rating"].to_numpy()
    y_pred_cf = cf.predict(test["user"].to_numpy(), test["item"].to_numpy())
    y_pred_hy = hybrid.predict(test["user"].to_numpy(), test["item"].to_numpy())

    print("\n=== Rating prediction ===")
    print(f"CF     RMSE={rmse(y_true, y_pred_cf):.4f}  MAE={mae(y_true, y_pred_cf):.4f}")
    print(f"Hybrid RMSE={rmse(y_true, y_pred_hy):.4f}  MAE={mae(y_true, y_pred_hy):.4f}")
    acc = relevance_accuracy(y_true, y_pred_hy, args.threshold) * 100
    print(f"Hybrid relevance accuracy @ {args.threshold}: {acc:.1f}%")

    # ---- ranking: full catalog
    print("\n=== Top-K ranking (full-catalog) ===")
    t0 = time.time()
    res = ranking_eval_vectorized(hybrid, train, test, ds.n_users, ds.n_items,
                                  k=args.k, threshold=args.threshold)
    print(f"Precision@{args.k}={res.precision_at_k:.4f}  "
          f"Recall@{args.k}={res.recall_at_k:.4f}  "
          f"({time.time() - t0:.2f}s)")

    # ---- ranking: sampled negatives (BPR/NCF protocol)
    print(f"\n=== Top-K ranking (sampled negatives: 1 + {args.n_negatives}, "
          f"threshold={args.rank_threshold}) ===")
    t0 = time.time()
    res_s = ranking_eval_sampled(hybrid, train, test, ds.n_users, ds.n_items,
                                 k=args.k, threshold=args.rank_threshold,
                                 n_negatives=args.n_negatives, seed=args.seed)
    print(f"Precision@{args.k}={res_s.precision_at_k:.4f}  "
          f"Recall@{args.k}={res_s.recall_at_k:.4f}  "
          f"({time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
