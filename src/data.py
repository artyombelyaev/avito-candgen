# Подготовка данных: компактные таблицы пар и объявлений вместо тяжёлого train (в pandas около 3,6 ГБ)
# и свойства объявлений для ранжировщика. item_id и query_id везде остаются строками.

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config, text

ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text", "item_category_id",
    "item_microcat_id", "item_price", "item_rating", "item_rating_reviews_count", "item_location_id",
    "item_latitude", "item_longitude", "item_is_phone_hidden", "item_is_message_forbidden",
]
_TEXT_COLS = {"search_query", "search_infm_params_text", "item_title_raw", "item_description_raw", "item_infm_params_text"}


# decimal переводим во float ещё в arrow, так быстрее.
def _read(path, columns=None) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    fields = []
    for field in table.schema:
        if pa.types.is_decimal(field.type):
            fields.append(pa.field(field.name, pa.float64()))
        elif pa.types.is_large_string(field.type):
            fields.append(pa.field(field.name, pa.string()))
        else:
            fields.append(field)
    df = table.cast(pa.schema(fields)).to_pandas()
    for col in df.columns:
        if col in _TEXT_COLS:
            df[col] = df[col].fillna("").astype(str)
    return df


def load_bench_queries() -> pd.DataFrame:
    return _read(config.BENCH_QUERIES_PATH)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# Пары запрос -> выбранное объявление без текстов объявлений.
# qn - нормализованный текст запроса, qid - номер запроса в порядке первого появления.
def build_pairs() -> pd.DataFrame:
    df = _read(config.TRAIN_PATH, config.SEARCH_COLS + ["item_id"])
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    df["qn"] = df.search_query.map(text.normalize)
    key = df[config.SEARCH_COLS[0]].astype(str)
    for col in config.SEARCH_COLS[1:]:
        key = key + "\x1f" + df[col].astype(str)
    df["qid"] = pd.factorize(key)[0].astype(np.int64)
    return df


# Уникальные объявления train и корпуса. Если объявление есть в обоих, берём версию корпуса.
def build_items() -> pd.DataFrame:
    parts, seen = [], set()
    for batch in pq.ParquetFile(config.TRAIN_PATH).iter_batches(batch_size=50_000, columns=ITEM_COLS):
        b = batch.to_pandas().drop_duplicates("item_id")
        b = b[~b.item_id.isin(seen)]
        seen.update(b.item_id)
        parts.append(b)
    train_items = pd.concat(parts, ignore_index=True)
    for col in ["item_price", "item_latitude", "item_longitude"]:
        train_items[col] = pd.to_numeric(train_items[col].astype(object), errors="coerce").astype(np.float64)
    for col in ["item_title_raw", "item_description_raw", "item_infm_params_text"]:
        train_items[col] = train_items[col].fillna("").astype(str)
    bench = _read(config.BENCH_ITEMS_PATH)
    bench_ids = set(bench.item_id)
    items = pd.concat([bench, train_items[~train_items.item_id.isin(bench_ids)]], ignore_index=True)
    items["in_bench"] = items.item_id.isin(bench_ids)
    items["in_train"] = items.item_id.isin(seen)
    return items


# Свойства объявления: помогают отличить выбранное от соседей с тем же текстом.
def build_item_static(it: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"item_id": it.item_id.values})
    out["item_location_id"] = it.item_location_id.values
    out["item_microcat_id"] = it.item_microcat_id.values
    out["item_category_id"] = it.item_category_id.values
    out["lat"] = it.item_latitude.values
    out["lon"] = it.item_longitude.values
    out["rating"] = it.item_rating.values
    out["log_reviews"] = np.log1p(it.item_rating_reviews_count.fillna(0).values)
    out["has_rating"] = it.item_rating.notna().values.astype(np.int8)
    price = it.item_price.values
    out["log_price"] = np.log1p(np.clip(price, 0, None))
    # цена 0 или 1 руб. - заглушка
    out["price_is_1"] = (price <= 1).astype(np.int8)
    median = it.groupby(["item_location_id", "item_microcat_id"]).item_price.transform("median")
    out["price_rel_med"] = np.log1p(np.clip(price, 0, None)) - np.log1p(np.clip(median.values, 0, None))
    out["phone_hidden"] = it.item_is_phone_hidden.astype(np.int8).values
    out["msg_forbidden"] = it.item_is_message_forbidden.astype(np.int8).values
    out["title_words"] = it.item_title_raw.str.split().str.len().fillna(0).values
    desc = it.item_description_raw.fillna("")
    out["desc_words"] = desc.str.split().str.len().fillna(0).values
    out["params_words"] = it.item_infm_params_text.str.split().str.len().fillna(0).values
    desc_key = desc.str.strip()
    out["desc_dup"] = np.log1p(desc_key.map(desc_key.value_counts()).values - 1)
    title_key = it.item_title_raw.map(text.normalize)
    out["title_dup"] = np.log1p(title_key.map(title_key.value_counts()).values - 1)
    out["emoji_share"] = desc.str.count("[\U0001F300-\U0001FAFF☀-➿]").values / (out["desc_words"].values + 1)
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    return out


def main():
    build_pairs().to_parquet(config.PAIRS_PATH, index=False)
    items = build_items()
    items.to_parquet(config.ITEMS_PATH, index=False)
    build_item_static(items).to_parquet(config.ITEM_STATIC_PATH, index=False)


if __name__ == "__main__":
    main()
