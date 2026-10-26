"""Limpieza, deduplicación, decontaminación y muestreo de corpus gn–es (funciones puras)."""

import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator

from guanani.normalize import normalize

MAX_WORDS = 200
MAX_RATIO = 3.0
# Subconjuntos de KuatIA de registro probablemente formal. Excluidos a propósito: fuentes de evaluación
# (flores-200, belele, jojajovai, commonvoice, americasnlp*, americasnli), Twitter, chat/afectivos y Biblia.
KUATIA_SUBSETS = ("madlad400", "culturax", "fineweb-2", "hplt-3", "FinePDF", "guarania-scraped", "multi-wiki-qa")
BT_TAG = "[BT] "

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
# KuatIA trae saltos de línea escapados como texto literal ("\\n"); se tratan como saltos reales.
_ESCAPES = {"\\n": "\n", "\\r": "\n", "\\t": " "}

# ponytail: reglas baratas complementarias a GlotLID (que decide la variedad): frenan el español/jopará muy
# cargado y el ruido numérico.
_ES_STOPWORDS = frozenset(
    "de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este "
    "porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos "
    "uno les ni contra otros ese eso ante ellos esto antes algunos qué unos yo otro otras otra él tanto esa "
    "estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros".split()
)


def passes_text_rules(sentence: str) -> bool:
    words = re.findall(r"[\w']+", sentence.lower())
    if not words:
        return False
    if sum(w in _ES_STOPWORDS for w in words) / len(words) > 0.25:
        return False
    return sum(any(c.isdigit() for c in w) for w in words) / len(words) <= 0.2


def norm_es(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def decont_key(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", normalize(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def clean_pairs(pairs: Iterable[dict], eval_keys: set[str]) -> tuple[list[dict], dict]:
    stats = dict.fromkeys(("in", "empty", "identical", "too_long", "ratio", "duplicate", "contaminated"), 0)
    seen, out = set(), []
    for p in pairs:
        stats["in"] += 1
        gn, es = normalize(p["gn"]), norm_es(p["es"])
        if not gn or not es:
            stats["empty"] += 1
            continue
        if gn.lower() == es.lower():
            stats["identical"] += 1
            continue
        ng, ne = len(gn.split()), len(es.split())
        if max(ng, ne) > MAX_WORDS:
            stats["too_long"] += 1
            continue
        if max(ng, ne) / min(ng, ne) > MAX_RATIO:
            stats["ratio"] += 1
            continue
        key = (gn.lower(), es.lower())
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        if decont_key(gn) in eval_keys or decont_key(es) in eval_keys:
            stats["contaminated"] += 1
            continue
        out.append({"gn": gn, "es": es, "source": p["source"]})
    return out, stats


def split_sentences(text: str) -> list[str]:
    for escaped, real in _ESCAPES.items():
        text = text.replace(escaped, real)
    return [re.sub(r"[ \t]+", " ", s).strip() for s in _SENT.split(text) if s.strip()]


def clean_sentences(
    docs: Iterable[tuple[str, str]],
    eval_keys: set[str],
    stats: Counter,
    seen: set[str],
    reject_lang: Callable[[str], str | None] | None = None,
    min_words: int = 5,
    max_words: int = 60,
) -> Iterator[tuple[str, str]]:
    # Orden: filtros baratos primero; GlotLID (el más caro) solo para frases nuevas y limpias.
    for source, doc in docs:
        for s in map(normalize, split_sentences(doc)):
            if not min_words <= len(s.split()) <= max_words:
                stats[f"{source}:longitud"] += 1
                continue
            if not passes_text_rules(s):
                stats[f"{source}:reglas"] += 1
                continue
            key = decont_key(s)
            if key in eval_keys:
                stats[f"{source}:contaminada"] += 1
                continue
            if key in seen:
                stats[f"{source}:duplicada"] += 1
                continue
            seen.add(key)
            label = reject_lang(s) if reject_lang else None
            if label:
                stats[f"{source}:lid:{label}"] += 1
                continue
            stats[f"{source}:aceptada"] += 1
            yield source, s


def reservoir(items: Iterable, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    sample = []
    for k, item in enumerate(items, 1):
        if len(sample) < n:
            sample.append(item)
        else:
            j = rng.randrange(k)
            if j < n:
                sample[j] = item
    return sample


def build_examples(parallel: list[dict], bt: list[dict] = (), bt_tag: bool = False) -> list[dict]:
    ex = []
    for p in parallel:
        ex.append({"src": p["gn"], "tgt": p["es"], "src_lang": "grn_Latn", "tgt_lang": "spa_Latn"})
        ex.append({"src": p["es"], "tgt": p["gn"], "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"})
    # Retrotraducción solo es→gn: el español es sintético, el guaraní es real.
    tag = BT_TAG if bt_tag else ""
    for p in bt:
        ex.append({"src": tag + p["es"], "tgt": p["gn"], "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"})
    return ex
