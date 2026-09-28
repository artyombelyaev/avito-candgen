# Источники кандидатов. Их списки объединяются в пул, из которого ранжировщик выбирает топ-50.
# Источники ошибаются по-разному, поэтому их несколько. Локация везде приоритет, а не фильтр.

import collections

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from . import config, data, text
from .split import allowed_items, context_pairs, load_part, local_items

PARTS = ["val", "rank", "bench"]
DENSE_MODELS = ["e5-base-ft", "e5-base-zs"]
RING_KM = 100


def cands_path(source: str, part: str):
    return config.CANDS / f"{source}_{part}.parquet"


# Списки кандидатов хранятся одной таблицей: query_id, item_id, score и rank (с нуля).
def save(source: str, part: str, lists: dict, scores: dict) -> None:
    query_ids, item_ids, values, ranks = [], [], [], []
    for qid, items in lists.items():
        query_ids.extend([qid] * len(items))
        item_ids.extend(items)
        ranks.extend(range(len(items)))
        values.extend(scores[qid])
    pd.DataFrame({"query_id": query_ids, "item_id": item_ids, "score": np.asarray(values, np.float32),
                  "rank": np.asarray(ranks, np.int32)}).to_parquet(cands_path(source, part), index=False)


def load(source: str, part: str) -> pd.DataFrame:
    return pd.read_parquet(cands_path(source, part))


def _split_words(s: str) -> list[str]:
    return s.split()


# BM25 по полям с весами, у каждого поля свои длины документов. Веса документов считаются
# заранее, и скоры батча запросов - одно умножение матриц.
class FieldBM25:
    def __init__(self, fields: dict, weights: dict, k1: float = 1.2, b: float = 0.75):
        self.vec = CountVectorizer(analyzer=_split_words, lowercase=False, dtype=np.float32)
        self.vec.fit(t for docs in fields.values() for t in docs)
        self.vocab = self.vec.vocabulary_
        self.fields = {name: self._bm25_weights(self.vec.transform(docs).tocsr().astype(np.float32), k1, b)
                       for name, docs in fields.items()}
        W = None
        for name, M in self.fields.items():
            W = M * weights[name] if W is None else W + M * weights[name]
        self.W = W.T.tocsr()

    @staticmethod
    def _bm25_weights(tf: sp.csr_matrix, k1: float, b: float) -> sp.csr_matrix:
        n = tf.shape[0]
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        avg = lengths.mean() if lengths.mean() > 0 else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        norm = k1 * (1 - b + b * lengths[tf.row] / avg)
        weights = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm)
        return sp.csr_matrix((weights.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    # вес формы - 1 / число форм слова в словаре, чтобы слово с двумя леммами не весило вдвое
    def query_matrix(self, queries) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for qi, words in enumerate(queries):
            acc = {}
            for forms in words:
                ids = [self.vocab[v] for v in dict.fromkeys(forms) if v in self.vocab]
                for j in ids:
                    acc[j] = acc.get(j, 0.0) + 1.0 / len(ids)
            for j, v in acc.items():
                rows.append(qi)
                cols.append(j)
                vals.append(v)
        return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(queries), len(self.vocab)))

    def score(self, Q: sp.csr_matrix) -> np.ndarray:
        return (Q @ self.W).toarray()


# Леммы полей кэшируются: лемматизация описаний идёт несколько минут.
# Из описания берём первые 300 слов, дальше обычно список всех услуг подряд.
LEMMA_CACHE = config.CACHE / "item_lemmas.parquet"


def _lemmatize(s: str, max_words: int | None = None) -> str:
    tokens = text.tokenize(s)[:max_words]
    return " ".join(v for t in tokens for v in text.lemma_variants(t) if v not in text.STOPWORDS)


def get_item_lemmas(item_ids) -> pd.DataFrame:
    ids = pd.Index(pd.unique(pd.Series(list(item_ids))), name="item_id")
    cache = pd.read_parquet(LEMMA_CACHE) if LEMMA_CACHE.exists() else pd.DataFrame(columns=["item_id", "title_l", "params_l", "desc_l"])
    missing = ids.difference(pd.Index(cache.item_id))
    if len(missing):
        items = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
        items = items[items.item_id.isin(set(missing))]
        new = pd.DataFrame({
            "item_id": items.item_id.values,
            "title_l": [_lemmatize(s) for s in items.item_title_raw],
            "params_l": [_lemmatize(s) for s in items.item_infm_params_text],
            "desc_l": [_lemmatize(s, 300) for s in items.item_description_raw],
        })
        cache = pd.concat([cache, new], ignore_index=True)
        cache.to_parquet(LEMMA_CACHE, index=False)
    return cache.set_index("item_id").loc[ids].reset_index()


# Описание нужно: без него в 17% пар нет ни одного общего слова. Больший вес заголовка вредил.
def build_index(corpus: pd.DataFrame) -> FieldBM25:
    lem = get_item_lemmas(corpus.item_id)
    return FieldBM25({"title": lem.title_l.tolist(), "params": lem.params_l.tolist(), "desc": lem.desc_l.tolist()},
                     weights={"title": 1.0, "params": 0.5, "desc": 1.0})


# alpha = inf - сначала вся локальная зона, alpha = 0 - без учёта локации.
# Возвращает топ-k с добивкой популярными, список глубины depth и его скоры.
def retrieve(queries: pd.DataFrame, index: FieldBM25, corpus: pd.DataFrame, ctx, alpha: float,
             k: int = config.TOP_K, depth: int = 1000, batch: int = 256):
    item_loc = corpus.item_location_id.values
    ids = corpus.item_id.values
    is_service = corpus.item_category_id.values == 114
    popularity = corpus.item_id.map(ctx.popularity).fillna(0).values + 1e-4 * np.log1p(corpus.item_rating_reviews_count.fillna(0).values)
    by_popularity = np.argsort(-popularity, kind="stable")
    Q = index.query_matrix([text.query_forms(q) for q in queries.search_query])
    top_k, top_depth, depth_scores = {}, {}, {}
    qids = queries.query_id.values
    for start in range(0, len(queries), batch):
        S = index.score(Q[start:start + batch])
        for j in range(S.shape[0]):
            qi = start + j
            row = queries.iloc[qi]
            score = S[j]
            allowed = is_service if row.search_category == 114 else np.ones(len(score), bool)
            local = ctx.local_mask(row.search_location_id, item_loc)
            if np.isinf(alpha):
                ordered = score + (score > 0) * local * 1e6
            else:
                ordered = score * (1.0 + alpha * local)
            ordered = np.where(allowed, ordered, -1.0)
            d = min(depth, int((ordered > 0).sum()))
            if d > 0:
                cand = np.argpartition(-ordered, d - 1)[:d] if d < len(ordered) else np.arange(len(ordered))
                cand = cand[np.argsort(-ordered[cand], kind="stable")]
                cand = cand[ordered[cand] > 0]
            else:
                cand = np.array([], dtype=int)
            top_depth[qids[qi]] = ids[cand].tolist()
            depth_scores[qids[qi]] = score[cand].astype(np.float32).tolist()
            answer = cand[:k].tolist()
            if len(answer) < k:
                answer = _fill(answer, k, local & allowed, allowed, by_popularity)
            top_k[qids[qi]] = ids[answer].tolist()
    return top_k, top_depth, depth_scores


def _fill(answer: list, k: int, local_allowed: np.ndarray, allowed: np.ndarray, by_popularity: np.ndarray) -> list:
    have = set(answer)
    for mask in (local_allowed, allowed):
        for i in by_popularity[mask[by_popularity]]:
            if len(answer) >= k:
                return answer
            if i not in have:
                answer.append(int(i))
                have.add(int(i))
    return answer


# Топ-50 BM25 по бенчмарку, им добиваются короткие ответы.
def bench_bm25_top50() -> dict:
    p = load_part("bench")
    return retrieve(p.queries, build_index(p.corpus), p.corpus, p.context, float("inf"))[0]


def local_first(idx: np.ndarray, score: np.ndarray, local: np.ndarray, depth: int):
    order = np.lexsort((-score, ~local[idx]))
    return idx[order][:depth], score[order][:depth]


# Второй список, без учёта локации, - для ответов в других городах и онлайн-услуг.
def bm25_candidates(part, p, pairs, index) -> None:
    _, local_lists, local_scores = retrieve(p.queries, index, p.corpus, p.context, float("inf"), depth=1000)
    save("bm25_local", part, local_lists, local_scores)
    _, global_lists, global_scores = retrieve(p.queries, index, p.corpus, p.context, 0.0, depth=300)
    save("bm25_global", part, global_lists, global_scores)


# Что выбирали в train по тому же нормализованному тексту: в той же локации и где угодно.
def memory_candidates(part, p, pairs, index, depth: int = 300) -> None:
    pairs = pairs[pairs.item_id.isin(p.item_pos)]
    by_text_loc = pairs.groupby(["qn", "search_location_id"]).item_id.value_counts()
    by_text = pairs.groupby("qn").item_id.value_counts()
    text_loc_keys = set(by_text_loc.index.droplevel(2))
    text_keys = set(by_text.index.droplevel(1))
    lists_tl, scores_tl, lists_t, scores_t = {}, {}, {}, {}
    for row in p.queries.itertuples():
        allowed = allowed_items(p, row)
        variants = [((row.qn, row.search_location_id), by_text_loc, text_loc_keys, lists_tl, scores_tl, False),
                    (row.qn, by_text, text_keys, lists_t, scores_t, True)]
        for key, counts, keys, lists, scores, use_local_first in variants:
            if key not in keys:
                continue
            c = counts.loc[key]
            idx = np.array([p.item_pos[i] for i in c.index])
            score = c.values.astype(np.float32)
            keep = allowed[idx]
            idx, score = idx[keep], score[keep]
            if use_local_first:
                idx, score = local_first(idx, score, local_items(p, row), depth)
            else:
                order = np.argsort(-score, kind="stable")[:depth]
                idx, score = idx[order], score[order]
            if len(idx):
                lists[row.query_id] = p.item_ids[idx].tolist()
                scores[row.query_id] = score.tolist()
    save("memory_loc", part, lists_tl, scores_tl)
    save("memory", part, lists_t, scores_t)


# Выборы по 30 похожим текстам из train (символьные 2-4-граммы ловят словоформы и опечатки).
def similar_queries_candidates(part, p, pairs, index, n_neighbors: int = 30, depth: int = 300) -> None:
    pairs = pairs[pairs.item_id.isin(p.item_pos)]
    by_text = pairs.groupby("qn").item_id.value_counts()
    texts = by_text.index.get_level_values(0).unique()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, dtype=np.float32)
    text_matrix = vectorizer.fit_transform(texts)
    query_matrix = vectorizer.transform(p.queries.qn)
    items_of_text = {t: (np.array([p.item_pos[i] for i in g.index.get_level_values(1)]), g.values.astype(np.float32))
                     for t, g in by_text.groupby(level=0)}
    lists, scores = {}, {}
    for start in range(0, len(p.queries), 512):
        sim = (query_matrix[start:start + 512] @ text_matrix.T).toarray()
        for j in range(sim.shape[0]):
            row = p.queries.iloc[start + j]
            neighbors = np.argpartition(-sim[j], n_neighbors)[:n_neighbors]
            acc = collections.defaultdict(float)
            for k in neighbors:
                if sim[j, k] <= 0:
                    continue
                idx, counts = items_of_text[texts[k]]
                for i, c in zip(idx, counts):
                    acc[i] += sim[j, k] * c
            if not acc:
                continue
            idx = np.fromiter(acc.keys(), dtype=np.int64)
            score = np.fromiter(acc.values(), dtype=np.float32)
            keep = allowed_items(p, row)[idx]
            idx, score = local_first(idx[keep], score[keep], local_items(p, row), depth)
            if len(idx):
                lists[row.query_id] = p.item_ids[idx].tolist()
                scores[row.query_id] = score.tolist()
    save("similar_queries", part, lists, scores)


# Символьные 3-5-граммы: опечатки (манекюр), склейки (автоюрист), сокращения и латиница.
def char_candidates(part, p, pairs, index, depth: int = 300, batch: int = 256) -> None:
    items = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_title_raw", "item_infm_params_text"]).set_index("item_id").loc[p.item_ids]
    docs = [text.item_text(t, prm, "", "tp") for t, prm in zip(items.item_title_raw, items.item_infm_params_text)]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.3, sublinear_tf=True, dtype=np.float32)
    doc_matrix = vectorizer.fit_transform(docs).T.tocsr()
    query_matrix = vectorizer.transform(p.queries.search_query.map(text.normalize))
    lists, scores = {}, {}
    for start in range(0, len(p.queries), batch):
        sim = (query_matrix[start:start + batch] @ doc_matrix).toarray()
        for j in range(sim.shape[0]):
            row = p.queries.iloc[start + j]
            score = np.where(allowed_items(p, row), sim[j], 0.0)
            # косинус не больше 1, поэтому +10 ставит локальную зону впереди остальных
            ordered = score + (score > 0) * local_items(p, row) * 10.0
            n = min(depth, int((ordered > 0).sum()))
            if n == 0:
                continue
            idx = np.argpartition(-ordered, n - 1)[:n]
            idx = idx[np.argsort(-ordered[idx], kind="stable")]
            lists[row.query_id] = p.item_ids[idx].tolist()
            scores[row.query_id] = score[idx].astype(np.float32).tolist()
    save("char", part, lists, scores)


def _classifier_words(s: str) -> list:
    query, filters = s.split(" | ", 1)
    return text.lemmas(query) + ["F_" + w for w in filters.split()]


def _query_part(s: str) -> str:
    return s.split(" | ", 1)[0]


# Классификатор запрос и фильтры -> микрокатегория. Микрокатегории реже 20 пар не учим, они шумные.
def train_microcat_classifier(pairs: pd.DataFrame, item_microcat: pd.Series):
    df = pairs.assign(mc=pairs.item_id.map(item_microcat)).dropna(subset=["mc"])
    df["x"] = df.qn + " | " + df.search_infm_params_text.str.lower()
    grouped = df.groupby(["x", "mc"]).size().rename("w").reset_index()
    counts = df.mc.value_counts()
    grouped = grouped[grouped.mc.isin(counts.index[counts >= 20])]
    word_vec = TfidfVectorizer(analyzer=_classifier_words, min_df=2, sublinear_tf=True, dtype=np.float32)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=150_000, sublinear_tf=True,
                               dtype=np.float32, preprocessor=_query_part)
    X = sp.hstack([word_vec.fit_transform(grouped.x), char_vec.fit_transform(grouped.x)]).tocsr()
    # логистическая регрессия через SGD: saga считалась бы десятки минут
    clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=15, tol=None, n_jobs=4, random_state=config.SEED)
    clf.fit(X, grouped.mc.astype(np.int64), sample_weight=np.log1p(grouped.w.values))
    return word_vec, char_vec, clf


# Популярное в трёх самых вероятных микрокатегориях: ответы без общих слов с запросом.
def microcat_candidates(part, p, pairs, index, depth: int = 200, top_mc: int = 3) -> None:
    items = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_microcat_id", "item_rating_reviews_count"]).set_index("item_id")
    word_vec, char_vec, clf = train_microcat_classifier(pairs, items.item_microcat_id)
    x = p.queries.qn + " | " + p.queries.search_infm_params_text.str.lower()
    proba = clf.predict_proba(sp.hstack([word_vec.transform(x), char_vec.transform(x)]).tocsr())
    classes = clf.classes_
    np.savez_compressed(config.CANDS / f"microcat_proba_{part}.npz", proba=proba.astype(np.float32),
                        classes=classes, query_id=p.queries.query_id.values)
    microcat = items.item_microcat_id.reindex(p.item_ids).values
    # число отзывов разбивает ничьи между объявлениями, которых никто не выбирал
    popularity = (pd.Series(p.item_ids).map(p.context.popularity).fillna(0).values
                  + 1e-3 * np.log1p(items.item_rating_reviews_count.reindex(p.item_ids).fillna(0).values))
    items_of_class = {c: np.flatnonzero(microcat == c) for c in classes}
    lists, scores = {}, {}
    for j, row in enumerate(p.queries.itertuples()):
        top = np.argsort(-proba[j])[:top_mc]
        idx = np.concatenate([items_of_class[classes[t]] for t in top])
        score = np.concatenate([np.full(len(items_of_class[classes[t]]), proba[j, t]) for t in top]) * (1 + popularity[idx])
        keep = allowed_items(p, row)[idx] & local_items(p, row)[idx]
        idx, score = idx[keep], score[keep]
        order = np.argsort(-score, kind="stable")[:depth]
        if len(order):
            lists[row.query_id] = p.item_ids[idx[order]].tolist()
            scores[row.query_id] = score[order].astype(np.float32).tolist()
    save("microcat", part, lists, scores)


def popular_candidates(part, p, pairs, index, depth: int = 100) -> None:
    popularity = pd.Series(p.item_ids).map(p.context.popularity).fillna(0).values
    lists, scores = {}, {}
    for row in p.queries.itertuples():
        idx = np.flatnonzero(allowed_items(p, row) & local_items(p, row) & (popularity > 0))
        idx = idx[np.argsort(-popularity[idx], kind="stable")[:depth]]
        if len(idx):
            lists[row.query_id] = p.item_ids[idx].tolist()
            scores[row.query_id] = popularity[idx].astype(np.float32).tolist()
    save("popular", part, lists, scores)


# Эмбеддинги из ноутбуков 01 и 02: items_tpd.npy (заголовок, параметры и описание),
# q_<часть>_qf.npy (запрос с фильтрами), items_order.parquet (порядок строк items_tpd.npy).
def load_embeddings(model: str, part: str):
    d = config.DENSE / model
    order = pd.read_parquet(d / "items_order.parquet").item_id.values
    items = np.load(d / "items_tpd.npy", mmap_mode="r")
    queries = np.load(d / f"q_{part}_qf.npy").astype(np.float32)
    return {iid: k for k, iid in enumerate(order)}, items, queries


def corpus_embeddings(model: str, part: str, p):
    row_of, items, queries = load_embeddings(model, part)
    return np.asarray(items[[row_of[i] for i in p.item_ids]], dtype=np.float32), queries


def dense_candidates(part, p, pairs, index, depth: int = 300, batch: int = 256) -> None:
    for model in DENSE_MODELS:
        corpus_emb, queries = corpus_embeddings(model, part, p)
        lists, scores = {}, {}
        for start in range(0, len(p.queries), batch):
            sim = queries[start:start + batch] @ corpus_emb.T
            for j in range(sim.shape[0]):
                row = p.queries.iloc[start + j]
                score = np.where(allowed_items(p, row), sim[j], -10.0)
                ordered = score + 10.0 * local_items(p, row)
                idx = np.argpartition(-ordered, depth)[:depth]
                idx = idx[np.argsort(-ordered[idx], kind="stable")]
                idx = idx[score[idx] > -10.0]
                lists[row.query_id] = p.item_ids[idx].tolist()
                scores[row.query_id] = score[idx].astype(np.float32).tolist()
        save(f"dense_{model}", part, lists, scores)


# Кольцо: до 100 км от центра локации запроса, но вне её локальной зоны, топ-50 по BM25 и по e5.
# Половина промахов - ответ в пригороде или соседнем городе, куда остальные источники не доходят.
def ring_candidates(part, p, pairs, index, depth: int = 50, batch: int = 256) -> None:
    static = pd.read_parquet(config.ITEM_STATIC_PATH, columns=["item_id", "lat", "lon"]).set_index("item_id").reindex(p.item_ids)
    lat, lon = static.lat.values, static.lon.values
    centroids = p.context.centroids
    Q = index.query_matrix([text.query_forms(q) for q in p.queries.search_query])
    corpus_emb, query_emb = corpus_embeddings("e5-base-ft", part, p)
    out = {"ring_bm25": ({}, {}), "ring_dense": ({}, {})}
    for start in range(0, len(p.queries), batch):
        bm25 = index.score(Q[start:start + batch])
        dense = query_emb[start:start + batch] @ corpus_emb.T
        for j in range(bm25.shape[0]):
            row = p.queries.iloc[start + j]
            loc = row.search_location_id
            if loc not in centroids.index:
                continue
            dist = data.haversine_km(centroids.at[loc, "lat"], centroids.at[loc, "lon"], lat, lon)
            ring = allowed_items(p, row) & ~local_items(p, row) & (dist <= RING_KM)
            if not ring.any():
                continue
            for name, score in (("ring_bm25", bm25[j]), ("ring_dense", dense[j])):
                idx = np.flatnonzero(ring & (score > 0) if name == "ring_bm25" else ring)
                if not len(idx):
                    continue
                idx = idx[np.argsort(-score[idx], kind="stable")[:depth]]
                lists, scores = out[name]
                lists[row.query_id] = p.item_ids[idx].tolist()
                scores[row.query_id] = score[idx].astype(np.float32).tolist()
    for name, (lists, scores) in out.items():
        save(name, part, lists, scores)


SOURCES = [bm25_candidates, memory_candidates, similar_queries_candidates, char_candidates,
           microcat_candidates, popular_candidates, dense_candidates, ring_candidates]


def main():
    for part in PARTS:
        p = load_part(part)
        pairs = context_pairs(part)
        index = build_index(p.corpus)
        for source in SOURCES:
            source(part, p, pairs, index)
        print(f"кандидаты {part}: готово", flush=True)


if __name__ == "__main__":
    main()
