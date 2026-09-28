# Пул кандидатов (около 1000 на запрос) и признаки пар запрос-кандидат для ранжировщика.

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, data, text
from .retrieval import DENSE_MODELS, PARTS, build_index, load, load_embeddings
from .split import PartData, context_pairs, load_part

# Порядок источников задаёт порядок строк и колонок, с которым обучался ранжировщик.
POOL_DEPTH = {"bm25_local": 500, "bm25_global": 100, "char": 200, "microcat": 150, "similar_queries": 100,
              "memory_loc": 999, "memory": 999, "popular": 50, "ring_bm25": 50, "ring_dense": 50,
              **{f"dense_{m}": 300 for m in DENSE_MODELS}}
# Их кандидаты остаются в пуле, но ранги не идут в признаки: на val это ничего не давало.
BEHAVIOR_SOURCES = ["similar_queries", "memory_loc", "memory", "popular"]

ITEM_FEATURES = ["rating", "log_reviews", "has_rating", "log_price", "price_is_1", "price_rel_med", "phone_hidden",
                 "msg_forbidden", "title_words", "desc_words", "params_words", "desc_dup", "title_dup", "emoji_share"]


def features_path(part: str):
    return config.ARTIFACTS / f"features_{part}.parquet"


# Таблица (query_id, item_id) с рангом и скором кандидата в каждом источнике (NaN, если его там нет).
def build_pool(part: str) -> pd.DataFrame:
    frames = []
    for source, depth in POOL_DEPTH.items():
        df = load(source, part)
        df = df[df["rank"] < depth][["query_id", "item_id", "rank", "score"]]
        frames.append(df.rename(columns={"rank": f"r_{source}", "score": f"s_{source}"}).set_index(["query_id", "item_id"]))
    pool = pd.concat(frames, axis=1, join="outer").reset_index()
    rank_cols = [c for c in pool.columns if c.startswith("r_")]
    pool["n_sources"] = pool[rank_cols].notna().sum(axis=1).astype(np.int8)
    pool = pool.drop(columns=[f"{k}_{s}" for s in BEHAVIOR_SOURCES for k in "rs"])
    for c in pool.columns:
        if c.startswith(("r_", "s_")):
            pool[c] = pool[c].astype(np.float32)
    return pool


# BM25 по каждому полю отдельно, косинус символьных n-грамм и доля слов запроса в каждом поле.
def lexical_features(pool: pd.DataFrame, p: PartData, ii: np.ndarray, groups: list) -> None:
    index = build_index(p.corpus)
    forms = [text.query_forms(x) for x in p.queries.search_query]
    Q = index.query_matrix(forms).tocsr()
    field_scores = {f: np.zeros(len(pool), np.float32) for f in index.fields}
    for j, rows in enumerate(groups):
        if len(rows) == 0:
            continue
        qv = Q[j].T
        for f, M in index.fields.items():
            field_scores[f][rows] = (M[ii[rows]] @ qv).toarray().ravel()
    for f, v in field_scores.items():
        pool[f"bm25_{f}"] = v

    items = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_title_raw", "item_infm_params_text"]).set_index("item_id").loc[p.item_ids]
    docs = [text.item_text(t, prm, "", "tp") for t, prm in zip(items.item_title_raw, items.item_infm_params_text)]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.3, sublinear_tf=True, dtype=np.float32)
    D = vectorizer.fit_transform(docs)
    Qc = vectorizer.transform(p.queries.search_query.map(text.normalize))
    char_cos = np.zeros(len(pool), np.float32)
    for j, rows in enumerate(groups):
        if len(rows):
            char_cos[rows] = (D[ii[rows]] @ Qc[j].T).toarray().ravel()
    pool["char_cos"] = char_cos
    del D, docs

    # ненулевой вес в матрице BM25 = слово есть в поле; множества слов заняли бы около 4 ГБ
    overlap = {f: np.full(len(pool), np.nan, np.float32) for f in index.fields}
    for j, rows in enumerate(groups):
        words = [[index.vocab[v] for v in variants if v in index.vocab] for variants in forms[j]]
        if len(rows) == 0 or not words:
            continue
        for f, M in index.fields.items():
            sub = M[ii[rows]]
            covered = np.zeros(len(rows), np.float32)
            for ids in words:
                if ids:
                    covered += sub[:, ids].getnnz(axis=1) > 0
            overlap[f][rows] = covered / len(words)
    for f, v in overlap.items():
        pool[f"ovl_{f}"] = v
    pool["all_in_title"] = (pool.ovl_title == 1).astype(np.int8)


def dense_features(pool: pd.DataFrame, p: PartData, part: str, qi: np.ndarray, ii: np.ndarray) -> None:
    for model in DENSE_MODELS:
        row_of, items, queries = load_embeddings(model, part)
        rows = np.array([row_of[i] for i in p.item_ids])[ii]
        cos = np.empty(len(pool), np.float32)
        for start in range(0, len(pool), 500_000):
            sl = slice(start, start + 500_000)
            cos[sl] = np.einsum("ij,ij->i", np.asarray(items[rows[sl]], dtype=np.float32), queries[qi[sl]])
        pool[f"cos_{model}"] = cos


def build(part: str) -> pd.DataFrame:
    p = load_part(part)
    pool = build_pool(part)
    qi = pool.query_id.map({q: j for j, q in enumerate(p.queries.query_id)}).values
    ii = pool.item_id.map(p.item_pos).values
    order = np.argsort(qi, kind="stable")
    bounds = np.searchsorted(qi[order], np.arange(len(p.queries) + 1))
    groups = [order[bounds[j]:bounds[j + 1]] for j in range(len(p.queries))]
    pool["label"] = 0
    if "answers" in p.queries.columns:
        answers = set((q, i) for q, items in zip(p.queries.query_id, p.queries.answers) for i in items)
        pool["label"] = np.fromiter(((q, i) in answers for q, i in zip(pool.query_id, pool.item_id)), dtype=np.int8, count=len(pool))

    lexical_features(pool, p, ii, groups)
    dense_features(pool, p, part, qi, ii)

    static = pd.read_parquet(config.ITEM_STATIC_PATH).set_index("item_id").loc[p.item_ids]
    pairs = context_pairs(part)
    item_loc = p.item_loc[ii]
    search_loc = p.queries.search_location_id.values[qi]
    pool["same_loc"] = (item_loc == search_loc).astype(np.int8)
    in_local = np.zeros(len(pool), np.int8)
    for j, rows in enumerate(groups):
        if len(rows):
            in_local[rows] = p.context.local_mask(p.queries.search_location_id.iat[j], item_loc[rows])
    pool["in_local"] = in_local
    centroids = p.context.centroids
    has_centroid = pd.Series(search_loc).isin(centroids.index).values
    lat = np.where(has_centroid, centroids.lat.reindex(search_loc).values, np.nan)
    lon = np.where(has_centroid, centroids.lon.reindex(search_loc).values, np.nan)
    pool["dist_km"] = data.haversine_km(lat, lon, static.lat.values[ii], static.lon.values[ii]).astype(np.float32)
    # P(локация объявления | локация поиска) по train, со сглаживанием для редких локаций
    loc_of = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_location_id"]).set_index("item_id").item_location_id
    loc_pairs = pairs.assign(il=pairs.item_id.map(loc_of)).groupby(["search_location_id", "il"]).size().to_dict()
    loc_total = pairs.groupby("search_location_id").size()
    num = np.array([loc_pairs.get(k, 0) for k in zip(search_loc, item_loc)], np.float32)
    den = pd.Series(search_loc).map(loc_total).fillna(0).values
    pool["p_loc"] = ((num + 0.1) / (den + 10)).astype(np.float32)

    saved = np.load(config.CANDS / f"microcat_proba_{part}.npz", allow_pickle=True)
    proba, classes = saved["proba"], saved["classes"]
    class_pos = {c: k for k, c in enumerate(classes)}
    k = np.array([class_pos.get(c, -1) for c in static.item_microcat_id.values[ii]])
    pool["p_microcat"] = np.where(k >= 0, proba[qi, np.clip(k, 0, None)], 0.0).astype(np.float32)
    class_rank = np.argsort(np.argsort(-proba, axis=1), axis=1)
    pool["microcat_rank"] = np.where(k >= 0, class_rank[qi, np.clip(k, 0, None)], 999).astype(np.int16)
    filters = [text.parse_query_filters(x) for x in p.queries.search_infm_params_text]
    item_params = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_infm_params_text"]).set_index("item_id").item_infm_params_text.loc[p.item_ids].values
    for key in ["vid", "tip"]:
        pool[f"f_{key}"] = np.array([text.item_matches(filters[a], item_params[b], key) for a, b in zip(qi, ii)], np.float32)
    wants_rating4 = np.array([bool(f["rating4"]) for f in filters])[qi]
    rating = static.rating.values[ii]
    pool["f_rating4_ok"] = np.where(wants_rating4, (rating >= 4) | np.isnan(rating), np.nan).astype(np.float32)

    for c in ITEM_FEATURES:
        pool[f"it_{c}"] = static[c].values[ii]

    seen_texts = set(pairs.qn)
    pool["q_len"] = np.array([len(text.query_forms(x)) for x in p.queries.search_query], np.int8)[qi]
    pool["q_has_filters"] = (p.queries.search_infm_params_text.values[qi] != "").astype(np.int8)
    pool["q_region"] = np.isin(search_loc, list(p.context.region_locs)).astype(np.int8)
    pool["q_text_seen"] = np.array([t in seen_texts for t in p.queries.qn], np.int8)[qi]
    pool["q_cat114"] = (p.queries.search_category.values[qi] == 114).astype(np.int8)
    return pool


def main():
    for part in PARTS:
        build(part).to_parquet(features_path(part), index=False)
        print(f"признаки {part}: готово", flush=True)


if __name__ == "__main__":
    main()
