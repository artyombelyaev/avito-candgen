# Проверка формата answer.csv: все запросы бенчмарка, в строке до 50 разных id из корпуса.
# Запуск: python check_submission.py answer.csv

import re
import sys

import pandas as pd
import pyarrow.parquet as pq

from src import config

ID_RE = re.compile(r"^[0-9a-f]{16}$")


def check(path, k: int = 50) -> bool:
    errors = []
    # dtype=str, чтобы id не превратились в числа
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["query_id", "answer"]:
        print(f"колонки {list(df.columns)}, ожидались ['query_id', 'answer']")
        return False
    bench_queries = set(pq.read_table(config.BENCH_QUERIES_PATH, columns=["query_id"]).column(0).to_pylist())
    corpus = set(pq.read_table(config.BENCH_ITEMS_PATH, columns=["item_id"]).column(0).to_pylist())
    if df.query_id.duplicated().any():
        errors.append("повторяющиеся query_id")
    if set(df.query_id) != bench_queries:
        errors.append("query_id не совпадают с benchmark_queries")
    for qid, answer in zip(df.query_id, df.answer):
        ids = answer.split(" ") if answer else []
        if answer != answer.strip() or "  " in answer:
            errors.append(f"{qid}: лишние пробелы")
        if len(ids) > k or len(set(ids)) != len(ids):
            errors.append(f"{qid}: больше {k} id или повторы")
        if any(not ID_RE.match(i) or i not in corpus for i in ids):
            errors.append(f"{qid}: id неверного формата или не из корпуса")
    for e in errors[:30]:
        print(e)
    if not errors:
        print(f"OK: {len(df)} строк, id в строке от {df.answer.str.split().str.len().min()} до {df.answer.str.split().str.len().max()}")
    return not errors


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else config.ANSWER_PATH) else 1)
