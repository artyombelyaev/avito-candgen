# Текст: нормализация и леммы для BM25, тексты для e5, разбор фильтров запроса.

import re
from functools import lru_cache

import pymorphy3

MORPH = pymorphy3.MorphAnalyzer()

# б/у и ж/д склеиваем в один токен, иначе получатся однобуквенные б и у
_SLASH_ABBR = re.compile(r"\b(\w)/(\w)\b")
_NON_WORD = re.compile(r"[^0-9a-zа-я]+")
_CYR = re.compile(r"[а-я]")
_LAT = re.compile(r"[a-z]")
_WORD = re.compile(r"\w+")

# Гомоглифы: латиница внутри русских слов (нa изгoтовлeние), без исправления такие слова не находятся.
_LAT2CYR = str.maketrans("aeopcxykmthb", "аеорсхукмтнв")
_CYR2LAT = str.maketrans("аеорсхукмтнв", "aeopcxykmthb")
_LAT_ONLY = re.compile(r"[dfgijlnqrsuvwz]")
_CYR_ONLY = re.compile(r"[бгдежзийлпфцчшщъыьэюя]")

# под и без оставляем: под ключ, без залога.
STOPWORDS = frozenset("и в во на с со по для к ко о об от из у за а но же ли бы то это как что или при".split())


# Алфавит слова определяет буква без двойника: И в CKИДКИ - кириллица, g в Сhаngаn - латиница.
# Иначе решает большинство.
def _fix_mixed_word(match: re.Match) -> str:
    word = match.group(0)
    n_cyr, n_lat = len(_CYR.findall(word)), len(_LAT.findall(word))
    if n_cyr == 0 or n_lat == 0:
        return word
    lat_only, cyr_only = bool(_LAT_ONLY.search(word)), bool(_CYR_ONLY.search(word))
    if cyr_only and not lat_only:
        return word.translate(_LAT2CYR)
    if lat_only and not cyr_only:
        return word.translate(_CYR2LAT)
    return word.translate(_LAT2CYR) if n_cyr >= n_lat else word.translate(_CYR2LAT)


def fix_homoglyphs(s: str) -> str:
    if not (_CYR.search(s) and _LAT.search(s)):
        return s
    return _WORD.sub(_fix_mixed_word, s)


def normalize(s: str) -> str:
    s = fix_homoglyphs(s.lower().replace("ё", "е"))
    s = _SLASH_ABBR.sub(r"\1\2", s)
    s = _NON_WORD.sub(" ", s)
    return " ".join(s.split())


def tokenize(s: str) -> list[str]:
    return normalize(s).split()


@lru_cache(maxsize=2_000_000)
def lemma(token: str) -> str:
    if not _CYR.search(token):
        return token
    return MORPH.parse(token)[0].normal_form.replace("ё", "е")


# До трёх нормальных форм токена: первый разбор pymorphy ошибается, и запрос электрик не находил
# Услуги электрика. Для слов вне словаря добавляем сам токен: роллеты -> роллета, роллет -> ролгод.
@lru_cache(maxsize=2_000_000)
def lemma_variants(token: str, max_variants: int = 3) -> tuple:
    if not _CYR.search(token):
        return (token,)
    variants = []
    for parse in MORPH.parse(token):
        form = parse.normal_form.replace("ё", "е")
        if form not in variants:
            variants.append(form)
        if len(variants) >= max_variants:
            break
    if not MORPH.word_is_known(token) and token not in variants:
        variants.append(token)
    return tuple(variants)


def lemmas(s: str) -> list[str]:
    return [t for t in (lemma(t) for t in tokenize(s)) if t not in STOPWORDS]


# Значимые слова запроса со всеми формами леммы.
def query_forms(q: str) -> list[tuple]:
    return [lemma_variants(t) for t in tokenize(q) if lemma(t) not in STOPWORDS]


# Тексты для e5: без лемматизации, убираем только гомоглифы, эмодзи и повторы пунктуации.
# Этот же код скопирован в ноутбуки 01 и 02.
DESC_WORDS = 120
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍⃣]+")
_REPEAT_PUNCT = re.compile(r"([^\w\s])\1+")
_SPACES = re.compile(r"\s+")


def clean(s: str) -> str:
    s = fix_homoglyphs((s or "").lower().replace("ё", "е"))
    s = _EMOJI.sub(" ", s)
    s = _REPEAT_PUNCT.sub(r"\1", s)
    return _SPACES.sub(" ", s).strip()


_VID = re.compile(r"Вид услуги (.+?)(?= Место оказания услуг| Тип услуги| Тип стоимости| Вид услуги|$)")
_TIP = re.compile(r"Тип услуги (?:автосервиса )?(.+?)(?= Место оказания услуг| Вид услуги| Тип услуги| Тип стоимости|$)")
_NAMES = re.compile(r"(?:Название услуги|Услуги) (.+?)(?= Начальная цена| Тип стоимости| Стоимость| Продолжительность|$)")


# Значение обрезаем на первом слове с заглавной буквы: с него начинается следующий ключ.
def _cut(v: str, max_words: int = 8) -> str:
    words = v.split()
    out = words[:1]
    for w in words[1:max_words]:
        if w[:1].isupper():
            break
        out.append(w)
    return " ".join(out)


# Из параметров берём вид, тип и названия услуг, остальное - одинаковый у всех шаблон.
def params_short(p: str, max_names: int = 8) -> str:
    p = p or ""
    vid = [_cut(v) for v in _VID.findall(p) if v and not v.startswith(("Место", "Тип"))]
    tip = [_cut(v) for v in _TIP.findall(p) if v and not v.startswith(("Место", "Вид"))]
    names = []
    for n in _NAMES.findall(p):
        n = _cut(n.strip())
        if n and n not in names and len(n) < 80:
            names.append(n)
        if len(names) >= max_names:
            break
    parts = []
    if vid:
        parts.append("вид услуги: " + ", ".join(dict.fromkeys(vid)))
    if tip:
        parts.append("тип услуги: " + ", ".join(dict.fromkeys(tip)))
    if names:
        parts.append("услуги: " + "; ".join(names))
    return clean(". ".join(parts))


# rep: tp - заголовок и параметры, tpd - ещё и начало описания (так кодировались объявления в e5).
def item_text(title: str, params: str, desc: str, rep: str = "tpd") -> str:
    parts = [clean(title)]
    if rep in ("tp", "tpd"):
        ps = params_short(params)
        if ps:
            parts.append(ps)
    if rep == "tpd":
        d = " ".join(clean(desc).split()[:DESC_WORDS])
        if d:
            parts.append(d)
    return ". ".join(parts)


def query_text(query: str, filters: str = "", with_filters: bool = False) -> str:
    q = clean(query)
    if with_filters and filters:
        q = q + ". " + clean(filters)
    return q


# Фильтры идут одной строкой без разделителей (Тип услуги Маникюр, педикюр Вид услуги Красота, здоровье),
# поэтому значение режем до следующего известного ключа.
_FILTER_KEYS = [
    "Вид услуги", "Тип услуги автосервиса", "Тип услуги", "Онлайн-запись", "Кто оказывает услуги",
    "Рейтинг пользователя", "Срочная услуга", "Где вы оказываете", "Ваши клиенты", "Опыт работы",
    "Гарантия", "Выезд за город", "Работа по договору", "Работа в праздники", "Специальность",
    "Предмет или", "Цена с НДС", "Доставка",
]
_NEXT_KEY = "(?=" + "|".join(" " + re.escape(k) for k in _FILTER_KEYS) + "|$)"
_FILTERS = {
    "vid": re.compile(r"Вид услуги (.+?)" + _NEXT_KEY),
    "tip": re.compile(r"Тип услуги(?! автосервиса) (.+?)" + _NEXT_KEY),
    "tip_auto": re.compile(r"Тип услуги автосервиса (.+?)" + _NEXT_KEY),
}
_ITEM_PREFIX = {"vid": "Вид услуги ", "tip": "Тип услуги ", "tip_auto": "Тип услуги автосервиса "}


def parse_query_filters(s: str) -> dict:
    out = {}
    for key, pattern in _FILTERS.items():
        out[key] = [v.strip() for v in pattern.findall(s) if v.strip() and not v.startswith(("Вид услуги", "Тип услуги"))]
    out["rating4"] = [True] if "Рейтинг пользователя 4 звезды" in s else []
    return out


# 1 - объявление подходит под фильтр, 0 - нет, NaN - такого фильтра в запросе нет.
def item_matches(filters: dict, item_params: str, key: str) -> float:
    values = filters.get(key) or []
    if not values:
        return float("nan")
    prefix = _ITEM_PREFIX[key]
    return float(any((prefix + v) in item_params for v in values))
