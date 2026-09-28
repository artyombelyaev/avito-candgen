# Пути и общие константы.

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Исходные parquet лежат в data/ или в папке из AVITO_DATA_DIR.
DATA_DIR = Path(os.environ.get("AVITO_DATA_DIR", ROOT / "data"))
TRAIN_PATH = DATA_DIR / "train.parquet"
BENCH_QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
BENCH_ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"

ARTIFACTS = ROOT / "artifacts"
CACHE = ARTIFACTS / "cache"
CANDS = ARTIFACTS / "cands"
DENSE = ARTIFACTS / "dense"
for _dir in (ARTIFACTS, CACHE, CANDS):
    _dir.mkdir(parents=True, exist_ok=True)

PAIRS_PATH = ARTIFACTS / "pairs.parquet"
ITEMS_PATH = ARTIFACTS / "items.parquet"
ITEM_STATIC_PATH = ARTIFACTS / "item_static.parquet"
SPLIT_PATH = ARTIFACTS / "split.parquet"
SPLIT_QUERIES_PATH = ARTIFACTS / "split_queries.parquet"
CORPORA_PATH = ARTIFACTS / "corpora.parquet"
RANKER_PATH = ARTIFACTS / "ranker.lgb"
ANSWER_PATH = ROOT / "answer.csv"

SEED = 42
TOP_K = 50

# Запрос в train - это пары с одинаковыми значениями этих колонок.
SEARCH_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
