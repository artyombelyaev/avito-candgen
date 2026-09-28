# Пары для дообучения e5 в ноутбуке 02: запрос -> выбранное объявление из base и два трудных негатива
# из топ-30 BM25. Ложные негативы отбрасываем: выбранные по тому же тексту, с тем же заголовком и той же
# микрокатегории. Иначе для аренда matiz негативом была бы Аренда авто Daewoo Matiz другого исполнителя.

import numpy as np
import pandas as pd

from . import config, text
from .retrieval import build_index, retrieve
from .split import build_context, context_pairs

OUT_PATH = config.ARTIFACTS / "ft_train.parquet"
N_TRAIN = 120_000
MAX_PER_TEXT = 5
NEG_DEPTH = 30
N_NEG = 2


def main():
    rng = np.random.default_rng(config.SEED)
    # base - train без запросов val и rank
    base = context_pairs("val").copy()
    items = pd.read_parquet(config.ITEMS_PATH, columns=["item_id", "item_title_raw", "item_location_id", "item_category_id",
                                                        "item_rating_reviews_count", "item_microcat_id"])
    microcat_of = dict(zip(items.item_id, items.item_microcat_id))

    base = base.sample(frac=1.0, random_state=config.SEED)
    pos = base.drop_duplicates(["qn", "item_id"])
    pos = pos.groupby("qn", sort=False).head(MAX_PER_TEXT)
    if len(pos) > N_TRAIN:
        pos = pos.sample(N_TRAIN, random_state=config.SEED)
    pos = pos.reset_index(drop=True)

    corpus = items[items.item_id.isin(set(base.item_id))].reset_index(drop=True)
    title_of = dict(zip(corpus.item_id, corpus.item_title_raw.map(text.normalize).values))
    # BM25 считаем один раз на каждый (текст, локация, категория)
    queries = pos.drop_duplicates(["qn", "search_location_id", "search_category"])[["qn", "search_query", "search_location_id", "search_category"]].copy()
    queries["query_id"] = [f"t{i}" for i in range(len(queries))]
    _, lists, _ = retrieve(queries, build_index(corpus), corpus, build_context(base), float("inf"), depth=100)
    list_of = {(a, b, c): lists.get(qid, []) for a, b, c, qid in
               zip(queries.qn, queries.search_location_id, queries.search_category, queries.query_id)}

    chosen_by_text = base.groupby("qn").item_id.apply(set)
    pos_titles_by_text = pos.assign(t=pos.item_id.map(title_of)).groupby("qn").t.apply(set)
    negs = []
    for r in pos.itertuples():
        chosen = chosen_by_text.get(r.qn, set())
        pos_titles = pos_titles_by_text.get(r.qn, set())
        pos_microcat = microcat_of.get(r.item_id)
        cand = [i for i in list_of[(r.qn, r.search_location_id, r.search_category)]
                if i not in chosen and title_of.get(i) not in pos_titles and microcat_of.get(i) != pos_microcat][:NEG_DEPTH]
        # если лексических соседей мало, добиваем случайными объявлениями другой микрокатегории
        while len(cand) < N_NEG:
            x = rng.choice(corpus.item_id.values)
            if microcat_of.get(x) != pos_microcat and x not in chosen:
                cand.append(x)
        negs.append(list(rng.choice(cand, N_NEG, replace=False)))
    negs = np.array(negs)
    pd.DataFrame({
        "search_query": pos.search_query.values, "search_infm_params_text": pos.search_infm_params_text.values,
        "qn": pos.qn.values, "search_location_id": pos.search_location_id.values,
        "pos": pos.item_id.values, **{f"neg{k + 1}": negs[:, k] for k in range(N_NEG)},
    }).to_parquet(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
