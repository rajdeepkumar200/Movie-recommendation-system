"""Flask app - the Netflix-style web UI on top of the hybrid recommender.

What's in here:
  - Catalog browsing / search / genre filter
  - Per-user profiles (stored as JSON in data/profiles/) with a like list
  - Cold-start recommendations from the liked items (averaged item factors)
  - Poster proxy: fetches from TMDB if a key is set, otherwise Wikipedia,
    caches the image bytes under data/poster_imgs/ so we don't hit the
    upstream every reload.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

# Allow running this file directly via `python web.py`.
_PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT / "src"))

from movie_recsys.collaborative import MatrixFactorization, MFConfig  # noqa: E402
from movie_recsys.data import (  # noqa: E402
    GENRE_COLS,
    build_item_features,
    build_user_features,
    load_ml100k,
    train_test_split_random,
)
from movie_recsys.hybrid import HybridRecommender  # noqa: E402
from movie_recsys.supervised import SupervisedConfig, SupervisedRecommender  # noqa: E402
from movie_recsys.svdpp import SVDpp, SVDppConfig  # noqa: E402

# -------- paths ---------------------------------------------------------------
DATA_DIR = _PROJECT / "data"
CACHE_PATH = DATA_DIR / "hybrid_model.pkl"
POSTER_CACHE_PATH = DATA_DIR / "posters.json"
IMG_CACHE_DIR = DATA_DIR / "poster_imgs"
PROFILES_DIR = DATA_DIR / "profiles"

# -------- posters -------------------------------------------------------------
# TMDB is higher quality but needs a key. Wikipedia is the fallback - works
# out of the box, but the `pilicense=any` flag is the magic bit that makes
# fair-use movie posters actually come back.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w342"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = "MovieRecsysDemo/1.0 (educational project)"

_poster_cache: dict[str, str | None] = {}
_poster_lock = threading.Lock()


def _load_poster_cache():
    global _poster_cache
    if POSTER_CACHE_PATH.exists():
        try:
            _poster_cache = json.loads(POSTER_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _poster_cache = {}


def _save_poster_cache():
    try:
        POSTER_CACHE_PATH.write_text(json.dumps(_poster_cache), encoding="utf-8")
    except Exception:
        pass


def _clean_title(title):
    """MovieLens titles look like 'Shawshank Redemption, The (1994)'.
    Pull out the year and move the article back to the front."""
    year = ""
    if "(" in title and title.endswith(")"):
        year = title.rsplit("(", 1)[1].rstrip(")").strip()
        title = title.rsplit("(", 1)[0].strip()
    for suffix in (", The", ", A", ", An", ", Les", ", La", ", Le"):
        if title.endswith(suffix):
            article = suffix[2:]
            title = f"{article} {title[: -len(suffix)]}"
            break
    return title.strip(), year


def _fetch_poster_tmdb(clean, year):
    if not TMDB_API_KEY:
        return None
    try:
        params = {"api_key": TMDB_API_KEY, "query": clean}
        if year:
            params["year"] = year
        r = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=4)
        if r.ok:
            results = r.json().get("results") or []
            if results and results[0].get("poster_path"):
                return f"{TMDB_IMG}{results[0]['poster_path']}"
    except Exception:
        pass
    return None


def _fetch_poster_wikipedia(clean, year):
    """Try a couple of search queries until one returns a page image.

    `pilicense=any` is the key - without it the API only returns
    free-licensed Commons images and ignores the (fair-use) posters on
    en.wikipedia.org/wikipedia/en/. Took me a while to find that one.
    """
    queries = []
    if year:
        queries.append(f"{clean} {year} film")
        queries.append(f"{clean} ({year} film)")
    queries.append(f"{clean} film")
    queries.append(clean)

    headers = {"User-Agent": WIKI_UA}
    for q in queries:
        try:
            params = {
                "action": "query",
                "format": "json",
                "prop": "pageimages",
                "piprop": "original|thumbnail",
                "pithumbsize": "400",
                "pilicense": "any",
                "generator": "search",
                "gsrsearch": q,
                "gsrlimit": "1",
            }
            r = requests.get(WIKI_API, params=params, headers=headers, timeout=5)
            if not r.ok:
                continue
            pages = ((r.json() or {}).get("query") or {}).get("pages") or {}
            for p in pages.values():
                thumb = (p.get("thumbnail") or {}).get("source")
                orig = (p.get("original") or {}).get("source")
                url = thumb or orig
                if url:
                    return url
        except Exception:
            continue
    return None


def fetch_poster(item_id, title):
    """Try TMDB, fall back to Wikipedia. Cache the result either way
    (including misses, so we don't repeatedly retry hopeless titles)."""
    key = str(item_id)
    if key in _poster_cache:
        return _poster_cache[key]

    clean, year = _clean_title(title)
    url = _fetch_poster_tmdb(clean, year) or _fetch_poster_wikipedia(clean, year)

    with _poster_lock:
        _poster_cache[key] = url
        if len(_poster_cache) % 25 == 0:
            _save_poster_cache()
    return url


# -------- app + global state --------------------------------------------------
app = Flask(__name__, template_folder="templates")

ds = None
train_df = None
hybrid_model: HybridRecommender | None = None

item_titles: dict[int, str] = {}
item_years: dict[int, str] = {}
item_genres: dict[int, list[str]] = {}
item_popularity: dict[int, int] = {}     # item_id -> rating count in train
top_popular_items: list[int] = []        # sorted by popularity


def _load_or_train_model():
    if CACHE_PATH.exists():
        try:
            print(f"Loading cached model from {CACHE_PATH}...")
            with CACHE_PATH.open("rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  cache load failed ({e}); retraining.")
    return _train_model()


def _train_model():
    """Same recipe as recommend.py - SVD++ + HGBT stacker, blended."""
    assert ds is not None and train_df is not None
    print("Training hybrid model...")

    cf = SVDpp(
        ds.n_users, ds.n_items,
        SVDppConfig(n_factors=80, n_epochs=30, lr=0.005, reg=0.05, seed=42, verbose=True),
    )
    t0 = time.time()
    cf.fit(train_df["user"].to_numpy(),
           train_df["item"].to_numpy(),
           train_df["rating"].to_numpy())
    print(f"CF trained in {time.time() - t0:.1f}s")

    user_feats = build_user_features(ds.users, ds.n_users)
    item_feats = build_item_features(ds.items, ds.n_items)
    sup = SupervisedRecommender(user_feats, item_feats, SupervisedConfig(seed=42))
    cf_pred = cf.predict(train_df["user"].to_numpy(), train_df["item"].to_numpy())
    t0 = time.time()
    sup.fit(train_df["user"].to_numpy(),
            train_df["item"].to_numpy(),
            train_df["rating"].to_numpy(),
            cf_pred)
    print(f"Supervised trained in {time.time() - t0:.1f}s")

    model = HybridRecommender(cf, sup, alpha=0.6)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as f:
        pickle.dump(model, f)
    print(f"Cached model to {CACHE_PATH}")
    return model


def _init_data():
    global ds, train_df, hybrid_model
    global item_titles, item_years, item_genres, item_popularity, top_popular_items

    print("Loading MovieLens 100K...")
    ds = load_ml100k(DATA_DIR)
    train_df, _ = train_test_split_random(ds.ratings, test_frac=0.2, seed=42)

    item_titles, item_years, item_genres = {}, {}, {}
    for _, row in ds.items.iterrows():
        i = int(row["item"])
        title = row["title"]
        item_titles[i] = title
        if "(" in title and title.endswith(")"):
            item_years[i] = title.rsplit("(", 1)[1].rstrip(")").strip()
        else:
            item_years[i] = ""
        item_genres[i] = [g for g in GENRE_COLS if row.get(g, 0)]

    # Popularity from the train split (rating count). Used by the
    # cold-start picker and to order search results sensibly.
    counts = train_df["item"].value_counts()
    item_popularity = {int(k): int(v) for k, v in counts.items()}
    top_popular_items = [i for i, _ in counts.items()]

    _load_poster_cache()
    hybrid_model = _load_or_train_model()

    if TMDB_API_KEY:
        print(f"[posters] TMDB key set - using TMDB first (Wikipedia fallback). "
              f"{len(_poster_cache)} cached.")
    else:
        print(f"[posters] Using Wikipedia (no key needed). "
              f"{len(_poster_cache)} cached. Set TMDB_API_KEY for nicer posters.")
    print("Ready.\n")


# -------- helpers -------------------------------------------------------------
def _movie_dict(i):
    return {
        "id": i,
        "title": item_titles.get(i, f"item#{i}"),
        "year": item_years.get(i, ""),
        "genres": item_genres.get(i, []),
        "popularity": item_popularity.get(i, 0),
    }


def _cold_start_scores(liked):
    """Average the CF item factors of the liked items into a pseudo p_u,
    then score every item. Standard cold-start trick for MF / SVD++."""
    assert hybrid_model is not None
    cf = hybrid_model.cf
    Q = cf.Q
    bi = cf.bi
    mu = cf.mu

    liked_arr = np.asarray(liked, dtype=np.int64)
    pseudo_p = Q[liked_arr].mean(axis=0)
    # Tiny user-bias prior so users with all-popular picks don't get
    # blown out by the item biases alone.
    pseudo_bu = float(bi[liked_arr].mean()) * 0.5
    scores = mu + pseudo_bu + bi + Q @ pseudo_p
    return np.clip(scores, 1.0, 5.0)


def _topk_with_metadata(scores, top, excluded=None, genre=""):
    """Mask out excluded items and an optional genre filter, return top-K."""
    assert ds is not None
    scores = scores.astype(np.float64).copy()
    if excluded:
        scores[list(excluded)] = -np.inf
    if genre:
        for i in range(ds.n_items):
            if genre not in item_genres.get(i, []):
                scores[i] = -np.inf

    top_n = min(top, ds.n_items)
    top_idx = np.argpartition(-scores, kth=top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    out = []
    for i in top_idx:
        i = int(i)
        if not np.isfinite(scores[i]):
            continue
        d = _movie_dict(i)
        d["predicted_rating"] = round(float(scores[i]), 2)
        out.append(d)
    return out


# -------- profile storage -----------------------------------------------------
_USERNAME_RE = re.compile(r"^[a-z0-9_]{3,20}$")


def _normalize_username(name):
    name = (name or "").strip().lower()
    return name if _USERNAME_RE.match(name) else None


def _profile_path(username):
    return PROFILES_DIR / f"{username}.json"


def _load_profile(username):
    p = _profile_path(username)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_profile(profile):
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile["updated_at"] = time.time()
    _profile_path(profile["username"]).write_text(
        json.dumps(profile, indent=2), encoding="utf-8",
    )


def _new_profile(username):
    now = time.time()
    return {"username": username, "liked": [], "created_at": now, "updated_at": now}


# -------- routes --------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", posters_enabled=True)


@app.route("/api/genres")
def api_genres():
    return jsonify(GENRE_COLS)


# ---- profiles
@app.route("/api/profile/<name>", methods=["GET"])
def api_profile_get(name):
    """Load a profile, auto-creating it if it doesn't exist yet."""
    username = _normalize_username(name)
    if not username:
        return jsonify({"error": "invalid username (3-20 chars: a-z, 0-9, _)"}), 400
    prof = _load_profile(username)
    created = False
    if prof is None:
        prof = _new_profile(username)
        _save_profile(prof)
        created = True
    return jsonify({"profile": prof, "created": created})


@app.route("/api/profile/<name>", methods=["PUT"])
def api_profile_replace(name):
    """Replace the liked list wholesale - used when the onboarding modal saves."""
    assert ds is not None
    username = _normalize_username(name)
    if not username:
        return jsonify({"error": "invalid username"}), 400
    payload = request.get_json(silent=True) or {}
    raw = payload.get("liked")
    if not isinstance(raw, list):
        return jsonify({"error": "liked must be a list of item ids"}), 400

    # Sanitize: int + in-range + dedupe (preserve order).
    liked = []
    seen = set()
    for x in raw:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= i < ds.n_items and i not in seen:
            seen.add(i)
            liked.append(i)

    prof = _load_profile(username) or _new_profile(username)
    prof["liked"] = liked
    _save_profile(prof)
    return jsonify({"profile": prof})


@app.route("/api/profile/<name>/like", methods=["POST"])
def api_profile_like(name):
    """Toggle like for a single item. Body: {"item_id": int}."""
    assert ds is not None
    username = _normalize_username(name)
    if not username:
        return jsonify({"error": "invalid username"}), 400
    payload = request.get_json(silent=True) or {}
    try:
        item_id = int(payload.get("item_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "missing item_id"}), 400
    if not (0 <= item_id < ds.n_items):
        return jsonify({"error": "item_id out of range"}), 400

    prof = _load_profile(username) or _new_profile(username)
    liked = list(prof.get("liked") or [])
    if item_id in liked:
        liked = [x for x in liked if x != item_id]
        action = "removed"
    else:
        liked.append(item_id)
        action = "added"
    prof["liked"] = liked
    _save_profile(prof)
    return jsonify({"profile": prof, "action": action})


@app.route("/api/profile/<name>/recommend", methods=["GET"])
def api_profile_recommend(name):
    """Top-N picks for this user based on their liked items (cold-start)."""
    username = _normalize_username(name)
    if not username:
        return jsonify({"error": "invalid username"}), 400
    prof = _load_profile(username)
    if not prof or not prof.get("liked"):
        return jsonify([])

    top = int(request.args.get("top", 20))
    genre = (request.args.get("genre") or "").strip()
    liked = list(prof["liked"])
    scores = _cold_start_scores(liked)
    return jsonify(_topk_with_metadata(scores, top, excluded=liked, genre=genre))


@app.route("/api/profile/<name>/likes", methods=["GET"])
def api_profile_likes(name):
    username = _normalize_username(name)
    if not username:
        return jsonify({"error": "invalid username"}), 400
    prof = _load_profile(username)
    if not prof:
        return jsonify([])
    return jsonify([_movie_dict(int(i)) for i in (prof.get("liked") or [])])


# ---- catalog + cold-start (anonymous)
@app.route("/api/movies")
def api_movies():
    assert ds is not None
    genre = (request.args.get("genre") or "").strip()
    q = (request.args.get("q") or "").lower().strip()
    limit = int(request.args.get("limit", 60))

    matched = []
    for i in range(ds.n_items):
        if genre and genre not in item_genres.get(i, []):
            continue
        if q and q not in item_titles.get(i, "").lower():
            continue
        matched.append(i)

    matched.sort(key=lambda i: -item_popularity.get(i, 0))
    return jsonify([_movie_dict(i) for i in matched[:limit]])


@app.route("/api/popular")
def api_popular():
    """Popular movies for the cold-start picker grid."""
    limit = int(request.args.get("limit", 60))
    genre = (request.args.get("genre") or "").strip()
    out = []
    for i in top_popular_items:
        if genre and genre not in item_genres.get(i, []):
            continue
        out.append(_movie_dict(i))
        if len(out) >= limit:
            break
    return jsonify(out)


@app.route("/api/cold_recommend", methods=["POST"])
def api_cold_recommend():
    """Stateless cold-start: client just sends a list of liked ids."""
    payload = request.get_json(silent=True) or {}
    liked = [int(x) for x in (payload.get("liked") or [])]
    top = int(payload.get("top", 20))
    genre = (payload.get("genre") or "").strip()
    if not liked:
        return jsonify([])
    scores = _cold_start_scores(liked)
    return jsonify(_topk_with_metadata(scores, top, excluded=liked, genre=genre))


# ---- poster proxy
@app.route("/api/img/<int:item_id>")
def api_img(item_id):
    """Proxy poster bytes through Flask.

    Two reasons not to let the browser hit the upstream directly:
      1. Wikipedia's fair-use images block hot-linking (Referer check).
      2. We get a local on-disk cache for free.
    """
    IMG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = IMG_CACHE_DIR / f"{item_id}.bin"

    if cache_path.exists():
        with cache_path.open("rb") as f:
            head = f.read(12)
        # Cheap content-type sniff from magic bytes.
        ct = "image/jpeg"
        if head.startswith(b"\x89PNG"):
            ct = "image/png"
        elif head[:6] in (b"GIF87a", b"GIF89a"):
            ct = "image/gif"
        elif head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            ct = "image/webp"
        elif b"<svg" in head.lower() or head.startswith(b"<?xml"):
            ct = "image/svg+xml"
        return send_file(cache_path, mimetype=ct, max_age=86400)

    title = item_titles.get(item_id, "")
    if not title:
        abort(404)
    url = fetch_poster(item_id, title)
    if not url:
        abort(404)

    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": WIKI_UA})
        if not r.ok or not r.content:
            abort(404)
        cache_path.write_bytes(r.content)
        ct = r.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        return Response(r.content, mimetype=ct,
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        abort(404)


# -------- entrypoint ----------------------------------------------------------
def main():
    _init_data()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
