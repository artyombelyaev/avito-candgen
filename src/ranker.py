# LightGBM lambdarank на запросах rank + val, ранняя остановка по 10% из них.

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .features import features_path

NON_FEATURES = {"query_id", "item_id", "label"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


# Признаки сразу в один массив float32: с копиями таблиц 8 ГБ памяти не хватало.
def train(tr: pd.DataFrame, feats: list[str], seed: int = config.SEED):
    # запросы без ответа в пуле ничему не учат
    keep = tr.groupby("query_id").label.transform("max").values > 0
    order = np.flatnonzero(keep)
    # строки одного запроса должны идти подряд
    order = order[np.argsort(tr.query_id.values[order], kind="stable")]
    qid = tr.query_id.values[order]
    y = tr.label.values[order].astype(np.float32)
    X = np.empty((len(order), len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        X[:, j] = tr[c].values[order]
    rng = np.random.default_rng(seed)
    unique_q = np.unique(qid)
    es_queries = set(rng.choice(unique_q, max(1, len(unique_q) // 10), replace=False))
    is_es = np.fromiter((x in es_queries for x in qid), dtype=bool, count=len(qid))
    sets = {}
    for name, m in [("fit", ~is_es), ("es", is_es)]:
        _, group_sizes = np.unique(qid[m], return_counts=True)
        sets[name] = lgb.Dataset(X[m], y[m], group=group_sizes, feature_name=feats, free_raw_data=True)
    del X
    params = dict(objective="lambdarank", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=seed, verbose=-1,
                  num_threads=6, metric="ndcg", eval_at=[50], lambdarank_truncation_level=100)
    sets["es"].reference = sets["fit"]
    return lgb.train(params, sets["fit"], num_boost_round=2000, valid_sets=[sets["es"]],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


# Скор кусками, чтобы не копировать всю таблицу признаков в float64.
def predict(model, df: pd.DataFrame, chunk: int = 1_000_000) -> np.ndarray:
    feats = model.feature_name()
    out = np.empty(len(df))
    for s in range(0, len(df), chunk):
        out[s:s + chunk] = model.predict(df[feats].iloc[s:s + chunk].to_numpy(np.float32), num_iteration=model.best_iteration)
    return out


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(features_path(p)) for p in ["rank", "val"]], ignore_index=True)
    model = train(tr, feature_cols(tr))
    model.save_model(str(config.RANKER_PATH), num_iteration=model.best_iteration)
    print(f"ранжировщик: {model.best_iteration} деревьев, {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
