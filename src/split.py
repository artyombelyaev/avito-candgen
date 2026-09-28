# Локальная валидация: делим train на val, rank и base так, чтобы отложенные запросы были похожи
# на бенчмарк, и собираем корпус каждой части. Статистики по train для val и rank берём только из base,
# для бенчмарка - из всего train, чтобы запрос никогда не попадал в свои же статистики.

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, data, text

N_VAL = 4000
N_RANK = 6000
ALL_RUSSIA = 621540

STATUS = ["0_unseen", "1_text", "2_text_loc", "3_full"]
FULL_KEY = ["qn", "search_location_id", "search_is_delivery_search", "search_infm_params_text", "search_category"]


# Регионы - локации поиска без объявлений: Москва и МО, вся Россия.
def region_locations(items: pd.DataFrame, pairs: pd.DataFrame) -> set:
    return set(pairs.search_location_id.unique()) - set(items.item_location_id.unique())


# Доли 16 ячеек у бенчмарка: насколько запрос совпадает с train x есть ли фильтры x регион ли локация.
def bench_targets(pairs: pd.DataFrame, bench_queries: pd.DataFrame, region_locs: set) -> pd.Series:
    texts = set(pairs.qn)
    text_locs = set(zip(pairs.qn, pairs.search_location_id))
    full_keys = set(map(tuple, pairs[FULL_KEY].values.tolist()))
    b = bench_queries[bench_queries.search_category == 114].copy()
    full = [tuple(r) in full_keys for r in b[FULL_KEY].values.tolist()]
    text_loc = [(a, c) in text_locs for a, c in zip(b.qn, b.search_location_id)]
    b["status"] = np.select([full, text_loc, b.qn.isin(texts).values], STATUS[3:0:-1], STATUS[0])
    b["has_params"] = b.search_infm_params_text != ""
    b["region"] = b.search_location_id.isin(region_locs)
    return b.groupby(["status", "has_params", "region"]).size() / len(b)


# Кандидаты в отложенные запросы: один случайный запрос на текст, как в бенчмарке.
# Пары запроса можно разделить с base, тогда получится полное совпадение.
def build_candidates(pairs: pd.DataFrame, region_locs: set, rng: np.random.Generator) -> pd.DataFrame:
    q = pairs.groupby("qid").agg(
        qn=("qn", "first"),
        search_location_id=("search_location_id", "first"),
        search_is_delivery_search=("search_is_delivery_search", "first"),
        search_infm_params_text=("search_infm_params_text", "first"),
        search_category=("search_category", "first"),
        n_rows=("row_id", "size"),
    ).reset_index()
    # как в бенчмарке: категория 114 без доставки
    eligible = q[(q.search_category == 114) & (q.search_is_delivery_search == 0)]
    n_text = q.groupby("qn").size()
    n_text_loc = q.groupby(["qn", "search_location_id"]).size()
    n_full = q.groupby(FULL_KEY).size()
    cand = eligible.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).drop_duplicates("qn")
    cand = cand.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    other_full = cand.set_index(FULL_KEY).index.map(n_full).values - 1
    other_text_loc = cand.set_index(["qn", "search_location_id"]).index.map(n_text_loc).values - 1
    other_text = cand.qn.map(n_text).values - 1
    cand["nat_status"] = np.select([other_full > 0, other_text_loc > 0, other_text > 0], STATUS[3:0:-1], STATUS[0])
    cand["splittable"] = cand.n_rows >= 2
    cand["has_params"] = cand.search_infm_params_text != ""
    cand["region"] = cand.search_location_id.isin(region_locs)
    return cand


# Размеры ячеек методом наибольшего остатка, в сумме ровно n.
def allocate(targets: pd.Series, n: int) -> pd.Series:
    raw = targets * n
    sizes = np.floor(raw).astype(int)
    order = (raw - sizes).sort_values(ascending=False).index[:n - sizes.sum()]
    sizes.loc[order] += 1
    return sizes


# Сначала ячейки полного совпадения: подходящих запросов для них мало.
def stratified_pick(cand: pd.DataFrame, used: np.ndarray, sizes: pd.Series) -> pd.DataFrame:
    need = sizes.copy()
    chosen = []
    free = ~used
    for idx in np.flatnonzero(free):
        r = cand.iloc[idx]
        cell = ("3_full", r.has_params, r.region)
        if need.get(cell, 0) <= 0:
            continue
        if r.nat_status == "3_full":
            chosen.append((idx, cell, False))
        elif r.splittable:
            chosen.append((idx, cell, True))
        else:
            continue
        need[cell] -= 1
        free[idx] = False
    for idx in np.flatnonzero(free):
        r = cand.iloc[idx]
        cell = (r.nat_status, r.has_params, r.region)
        if r.nat_status == "3_full" or need.get(cell, 0) <= 0:
            continue
        chosen.append((idx, cell, False))
        need[cell] -= 1
        free[idx] = False
    out = cand.iloc[[c[0] for c in chosen]].copy()
    out["row_split"] = [c[2] for c in chosen]
    used[[c[0] for c in chosen]] = True
    return out


# При разделении пара уходит в часть с вероятностью 0.5, но хотя бы одна остаётся с каждой стороны.
def assign_rows(pairs: pd.DataFrame, picked: pd.DataFrame, part: str, rng: np.random.Generator) -> pd.DataFrame:
    rows_of_qid = pairs[pairs.qid.isin(picked.qid)][["row_id", "qid"]].groupby("qid").row_id.apply(np.array)
    out = []
    for i, r in enumerate(picked.itertuples()):
        row_ids = rows_of_qid[r.qid]
        if r.row_split:
            mask = rng.random(len(row_ids)) < 0.5
            if mask.all() or not mask.any():
                mask = np.zeros(len(row_ids), bool)
                mask[rng.integers(len(row_ids))] = True
            row_ids = row_ids[mask]
        out.append(pd.DataFrame({"row_id": row_ids, "part": part, "query_id": f"{part}_{i:05d}"}))
    return pd.concat(out, ignore_index=True)


def main():
    rng = np.random.default_rng(config.SEED)
    pairs = pd.read_parquet(config.PAIRS_PATH)
    items = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_location_id", "in_bench"])
    bench_queries = data.load_bench_queries()
    bench_queries["qn"] = bench_queries.search_query.map(text.normalize)
    region_locs = region_locations(items, pairs)

    targets = bench_targets(pairs, bench_queries, region_locs)
    cand = build_candidates(pairs, region_locs, rng)
    used = np.zeros(len(cand), bool)
    picked_val = stratified_pick(cand, used, allocate(targets, N_VAL))
    picked_rank = stratified_pick(cand, used, allocate(targets, N_RANK))
    held = pd.concat([assign_rows(pairs, picked_val, "val", rng), assign_rows(pairs, picked_rank, "rank", rng)], ignore_index=True)
    split = pairs[["row_id"]].merge(held, on="row_id", how="left")
    split["part"] = split.part.fillna("base")
    split["query_id"] = split.query_id.fillna("")
    split.to_parquet(config.SPLIT_PATH, index=False)

    queries = held.merge(pairs, on="row_id").groupby("query_id").agg(
        part=("part", "first"), **{c: (c, "first") for c in config.SEARCH_COLS}, qn=("qn", "first"),
        answers=("item_id", lambda x: sorted(set(x))),
    ).reset_index()
    queries.to_parquet(config.SPLIT_QUERIES_PATH, index=False)

    # в benchmark_items только около 7% ответов на отложенные запросы,
    # поэтому корпус части - benchmark_items плюс её ответы
    corpora = items[["item_id", "in_bench"]].copy()
    for part in ["val", "rank"]:
        answers = set(i for a in queries[queries.part == part].answers for i in a)
        corpora[f"in_{part}"] = corpora.in_bench | corpora.item_id.isin(answers)
    corpora[corpora[["in_bench", "in_val", "in_rank"]].any(axis=1)].reset_index(drop=True).to_parquet(config.CORPORA_PATH, index=False)


def load_queries(part: str) -> pd.DataFrame:
    if part == "bench":
        return data.load_bench_queries()
    queries = pd.read_parquet(config.SPLIT_QUERIES_PATH)
    return queries[queries.part == part].reset_index(drop=True)


CORPUS_COLS = ["item_id", "item_location_id", "item_category_id", "item_latitude", "item_longitude", "item_rating_reviews_count"]


def load_corpus(part: str) -> pd.DataFrame:
    items = pd.read_parquet(config.ITEMS_PATH, columns=CORPUS_COLS + ["in_bench"])
    if part == "bench":
        ids = set(items.item_id[items.in_bench])
    else:
        corpora = pd.read_parquet(config.CORPORA_PATH)
        ids = set(corpora.item_id[corpora[f"in_{part}"]])
    return items[items.item_id.isin(ids)].drop(columns="in_bench").reset_index(drop=True)


def context_pairs(part: str) -> pd.DataFrame:
    pairs = pd.read_parquet(config.PAIRS_PATH)
    if part == "bench":
        return pairs
    split = pd.read_parquet(config.SPLIT_PATH, columns=["part"])
    return pairs[split.part.values == "base"]


# Статистики по train для поиска: локальные зоны, центры локаций, популярность объявлений.
@dataclass
class Context:
    region_map: dict            # регион -> локации его зоны, None - вся Россия
    centroids: pd.DataFrame     # локация -> медиана координат
    popularity: pd.Series       # item_id -> сколько раз выбирали
    region_locs: set

    # Локальная зона: для города та же локация (там 93% выборов), для региона - локации,
    # дающие 90% выборов, для всей России - всё.
    def local_mask(self, search_loc: int, item_loc: np.ndarray) -> np.ndarray:
        if search_loc in self.region_map:
            area = self.region_map[search_loc]
            if area is None:
                return np.ones(len(item_loc), bool)
            return np.isin(item_loc, list(area))
        return item_loc == search_loc


def build_context(pairs: pd.DataFrame, cover: float = 0.9) -> Context:
    items = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_location_id", "item_latitude", "item_longitude"])
    region_locs = set(pairs.search_location_id.unique()) - set(items.item_location_id.unique())
    item_loc = items.set_index("item_id").item_location_id
    region_map = {}
    for loc in region_locs:
        if loc == ALL_RUSSIA:
            region_map[loc] = None
            continue
        shares = pairs.loc[pairs.search_location_id == loc, "item_id"].map(item_loc).value_counts(normalize=True)
        region_map[loc] = set(shares.index[: int(np.searchsorted(shares.cumsum().values, cover)) + 1])
    centroids = items.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
    centroids.columns = ["lat", "lon"]
    # центр региона - медиана координат выбранных по нему объявлений
    by_id = items.set_index("item_id")
    for loc in region_locs:
        ids = pairs.loc[pairs.search_location_id == loc, "item_id"]
        if len(ids) and loc != ALL_RUSSIA:
            centroids.loc[loc] = by_id.loc[ids, ["item_latitude", "item_longitude"]].median().values
    return Context(region_map=region_map, centroids=centroids, popularity=pairs.item_id.value_counts(), region_locs=region_locs)


# Всё, что нужно источникам кандидатов про часть.
@dataclass
class PartData:
    queries: pd.DataFrame
    corpus: pd.DataFrame
    context: Context
    item_ids: np.ndarray
    item_pos: dict
    item_loc: np.ndarray
    is_service: np.ndarray


def load_part(part: str) -> PartData:
    queries = load_queries(part)
    queries["qn"] = queries.search_query.map(text.normalize)
    corpus = load_corpus(part)
    ids = corpus.item_id.values
    return PartData(queries=queries, corpus=corpus, context=build_context(context_pairs(part)), item_ids=ids,
                    item_pos={iid: i for i, iid in enumerate(ids)}, item_loc=corpus.item_location_id.values,
                    is_service=corpus.item_category_id.values == 114)


# Категория - единственный жёсткий фильтр: у запросов категории 114 так в 99.997% пар.
def allowed_items(p: PartData, row) -> np.ndarray:
    return p.is_service if row.search_category == 114 else np.ones(len(p.item_ids), bool)


def local_items(p: PartData, row) -> np.ndarray:
    return p.context.local_mask(row.search_location_id, p.item_loc)


if __name__ == "__main__":
    main()
