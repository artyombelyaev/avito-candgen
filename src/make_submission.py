# От исходных parquet до answer.csv. Каждый шаг - отдельный процесс, чтобы освобождалась память,
# готовые шаги пропускаются. Эмбеддинги e5 должны лежать в artifacts/dense/.

import subprocess
import sys

import lightgbm as lgb
import pandas as pd

from . import config
from .features import POOL_DEPTH, features_path
from .ranker import predict
from .retrieval import DENSE_MODELS, PARTS, bench_bm25_top50, cands_path

STEPS = [
    ("src.data", [config.PAIRS_PATH, config.ITEMS_PATH, config.ITEM_STATIC_PATH]),
    ("src.split", [config.SPLIT_PATH, config.SPLIT_QUERIES_PATH, config.CORPORA_PATH]),
    ("src.retrieval", [cands_path(s, p) for s in POOL_DEPTH for p in PARTS]),
    ("src.features", [features_path(p) for p in PARTS]),
    ("src.ranker", [config.RANKER_PATH]),
]


def main():
    for model in DENSE_MODELS:
        if not (config.DENSE / model / "items_tpd.npy").exists():
            sys.exit(f"нет эмбеддингов artifacts/dense/{model}/: скачайте их из релиза или посчитайте ноутбуками 01 и 02")
    for module, outputs in STEPS:
        if not all(path.exists() for path in outputs):
            print(f"> python -m {module}", flush=True)
            subprocess.run([sys.executable, "-m", module], check=True, cwd=config.ROOT)

    df = pd.read_parquet(features_path("bench"))
    df["score"] = predict(lgb.Booster(model_file=str(config.RANKER_PATH)), df)
    df = df.sort_values(["query_id", "score"], ascending=[True, False])
    top = {q: g.item_id.head(config.TOP_K).tolist() for q, g in df.groupby("query_id", sort=False)}
    del df
    # если в пуле запроса меньше 50 кандидатов, добиваем списком BM25 без повторов
    fallback = bench_bm25_top50()
    queries = pd.read_parquet(config.BENCH_QUERIES_PATH, columns=["query_id"]).query_id
    answers = []
    for qid in queries:
        answer = top.get(qid, [])
        if len(answer) < config.TOP_K:
            seen = set(answer)
            answer = answer + [i for i in fallback[qid] if i not in seen][:config.TOP_K - len(answer)]
        answers.append(" ".join(answer))
    pd.DataFrame({"query_id": queries, "answer": answers}).to_csv(config.ANSWER_PATH, index=False)
    subprocess.run([sys.executable, "check_submission.py", str(config.ANSWER_PATH)], check=True, cwd=config.ROOT)


if __name__ == "__main__":
    main()
