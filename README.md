# Кандидатогенерация для поиска услуг Авито

По запросу (текст, локация, фильтры, категория) нужно выбрать до 50 объявлений из корпуса в 189 тысяч.
Метрика - Recall@50, усреднённый по запросам.

Результат: Recall@50 0,9089 на лидерборде. На своей валидации (4000 отложенных запросов из train) та же схема
с ранжировщиком, обученным только на rank, даёт 0,953. Финальный ранжировщик я обучил на rank + val, поэтому
на val его уже не проверить.

## Как устроено решение

1. Валидация. Отложил из train 4000 запросов для оценки (val) и 6000 для обучения ранжировщика (rank) и собрал их так же,
   как, судя по данным, собран бенчмарк. Всё, что считается по train, для них беру только из оставшейся части.
2. Текст. Исправляю латинские буквы внутри русских слов и индексирую все формы слова по pymorphy3: первый разбор часто
   ошибается.
3. Кандидаты. Собираю около 1000 объявлений на запрос из нескольких источников: BM25, символьные n-граммы, популярное
   в предсказанных микрокатегориях, то, что выбирали по похожим запросам, и e5-base. Категория - жёсткий фильтр,
   локация - приоритет.
4. Дообучение e5. Дообучил e5-base в Colab на парах запрос -> выбранное объявление с трудными негативами из BM25.
   После этого он в одиночку находит больше BM25: Recall@50 0,856 против 0,813 на val.
5. Кольцо. Половина промахов была вне локальной зоны запроса, чаще всего в пригородах и соседних городах,
   поэтому добавил кандидатов в радиусе 100 км. На валидации это дало +1,1 п.п., а на лидерборде почти ничего
   (0,9083 -> 0,9089).
6. Ранжировщик. LightGBM lambdarank выбирает топ-50 по рангам источников, BM25 по полям, косинусам e5, локации,
   микрокатегории, фильтрам и свойствам объявления. Популярность и память по train как признаки попробовал,
   на val они ничего не дали, и я их убрал.

## Как запустить

Нужны Python 3.13 и 8 ГБ памяти, на macOS ещё `brew install libomp`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Исходные `train.parquet`, `benchmark_queries.parquet` и `benchmark_items.parquet` положите в `data/`
(или укажите папку в переменной `AVITO_DATA_DIR`).

1. Подготовка, около получаса:
   ```bash
   .venv/bin/python -m src.data
   .venv/bin/python -m src.split
   .venv/bin/python -m src.hardneg
   ```
2. Эмбеддинги в Colab на T4. В `MyDrive/avito_dataset/` положите три исходных parquet и из `artifacts/` файлы
   `split.parquet`, `split_queries.parquet`, `corpora.parquet`, `ft_train.parquet`. Запустите
   `notebooks/01_dense_zero_shot.ipynb` (около часа) и `notebooks/02_finetune_biencoder.ipynb` (около двух часов).
   Каждый в конце скачивает архив, оба распакуйте в `artifacts/dense/`.
3. Ответ, около 35 минут:
   ```bash
   .venv/bin/python -m src.make_submission
   ```
   Пересобирает `answer.csv` и проверяет его формат.

Обучение на GPU не детерминировано, поэтому точно тот же `answer.csv` получается только с моими эмбеддингами.
Они лежат в релизе репозитория: скачайте `dense_embeddings.zip` в корень репозитория, распакуйте и сразу запускайте шаг 3.
Подготовку данных он сделает сам, Colab и hardneg тогда не нужны, всё займёт меньше часа.

```bash
mkdir -p artifacts/dense && unzip dense_embeddings.zip -d artifacts/dense
.venv/bin/python -m src.make_submission
```

## Структура

- `src/config.py` - пути и константы.
- `src/data.py` - компактные таблицы пар и объявлений из исходных parquet, свойства объявлений.
- `src/text.py` - нормализация и леммы, тексты для e5, разбор фильтров запроса.
- `src/split.py` - части val, rank и base, их корпуса и статистики train, доступные каждой части.
- `src/retrieval.py` - источники кандидатов, включая e5 и кольцо.
- `src/features.py` - пул кандидатов и признаки для ранжировщика.
- `src/ranker.py` - обучение LightGBM.
- `src/make_submission.py` - весь путь до `answer.csv`.
- `src/hardneg.py` - обучающие пары для ноутбука 02.
- `notebooks/01_dense_zero_shot.ipynb` - сравнение энкодеров и эмбеддинги e5-base без дообучения.
- `notebooks/02_finetune_biencoder.ipynb` - дообучение e5-base и его эмбеддинги.
- `check_submission.py` - проверка формата ответа.
- `answer.csv` - отправленный ответ.

## Библиотеки и модели

- [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org), [SciPy](https://scipy.org), [scikit-learn](https://scikit-learn.org) - BSD-3-Clause
- [PyArrow](https://arrow.apache.org) - Apache-2.0
- [pymorphy3](https://github.com/no-plagiarism/pymorphy3) и словари `pymorphy3-dicts-ru` - MIT
- [LightGBM](https://github.com/microsoft/LightGBM) - MIT
- в Colab: [PyTorch](https://pytorch.org) (BSD-3-Clause), [sentence-transformers](https://github.com/UKPLab/sentence-transformers), [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets), [accelerate](https://github.com/huggingface/accelerate) (Apache-2.0)
- [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) - MIT, основной энкодер
- [sergeyzh/rubert-tiny-turbo](https://huggingface.co/sergeyzh/rubert-tiny-turbo) (MIT) и [deepvk/USER-bge-m3](https://huggingface.co/deepvk/USER-bge-m3) (Apache-2.0) - только для сравнения в ноутбуке 01

Внешние API я не использовал.
