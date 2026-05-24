"""Time the naive vs vectorized ranking-eval loops.

Just trains a small MF model and runs both eval implementations on a
subsample of test users. The naive version is O(U * I * predict_call),
so it gets slow fast - that's why we subsample.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make this runnable as `python scripts/benchmark.py` without installing.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from movie_recsys.collaborative import MatrixFactorization, MFConfig  # noqa: E402
from movie_recsys.data import load_ml100k, train_test_split_by_time  # noqa: E402
from movie_recsys.evaluate import ranking_eval_naive, ranking_eval_vectorized  # noqa: E402


def main():
    ds = load_ml100k("data")
    train, test = train_test_split_by_time(ds.ratings, test_frac=0.2, seed=42)

    # Small model is fine - this is timing the eval loop, not the model.
    cf = MatrixFactorization(
        ds.n_users, ds.n_items,
        MFConfig(n_factors=32, n_epochs=5, batch_size=8192, verbose=False),
    )
    cf.fit(train["user"].to_numpy(), train["item"].to_numpy(), train["rating"].to_numpy())

    # Subsample test users so the naive loop finishes this decade.
    rng = np.random.default_rng(0)
    sample_users = rng.choice(test["user"].unique(), size=200, replace=False)
    test_sample = test[test["user"].isin(sample_users)].reset_index(drop=True)

    t0 = time.time()
    naive = ranking_eval_naive(cf, train, test_sample, ds.n_users, ds.n_items, k=10)
    t_naive = time.time() - t0

    t0 = time.time()
    vec = ranking_eval_vectorized(cf, train, test_sample, ds.n_users, ds.n_items, k=10)
    t_vec = time.time() - t0

    speedup = t_naive / max(t_vec, 1e-9)
    print(f"naive      : {t_naive:7.2f}s  P@10={naive.precision_at_k:.4f}")
    print(f"vectorized : {t_vec:7.2f}s  P@10={vec.precision_at_k:.4f}")
    print(f"speedup    : {speedup:.1f}x  (~{(1 - t_vec / t_naive) * 100:.0f}% faster)")


if __name__ == "__main__":
    main()
