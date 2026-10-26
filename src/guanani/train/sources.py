"""Lectura de corpus de entrenamiento y construcción de data/train/."""

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from guanani import data
from guanani.train.corpus import KUATIA_SUBSETS, clean_pairs, decont_key

KUATIA_TEXT_FIELD = "text"
AGUSTIN_COLUMNS = (("gn", "es"), ("guarani", "spanish"), ("grn", "spa"))


def train_dir() -> Path:
    return data.data_dir() / "train"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def eval_keys() -> set[str]:
    texts = []
    for split in ("dev", "devtest"):
        texts += [t for r in data.load_flores(split) for t in (r["gn"], r["es"])]
    for r in data.load_jojajovai("test") + data.load_jojajovai_annotated():
        texts += [r["gn"], r["es"]]
    texts += [r["gn"] for r in data.load_common_voice("test")]
    return {k for k in map(decont_key, texts) if k}


def load_americasnlp() -> list[dict]:
    d = data.data_dir() / "americasnlp"
    gn = data._require(d / "train.gn", "americasnlp").read_text(encoding="utf-8").splitlines()
    es = data._require(d / "train.es", "americasnlp").read_text(encoding="utf-8").splitlines()
    if len(gn) != len(es):
        raise ValueError(f"AmericasNLP: train.gn ({len(gn)}) y train.es ({len(es)}) no están alineados")
    return [{"gn": g, "es": e, "source": "americasnlp"} for g, e in zip(gn, es)]


def load_agustin() -> tuple[list[dict], str | None]:
    p = data.data_dir() / "agustin" / "train.parquet"
    if not p.exists():
        return [], "no descargado"
    import pandas as pd

    df = pd.read_parquet(p)
    cols = {c.lower(): c for c in df.columns}
    for g, e in AGUSTIN_COLUMNS:
        if g in cols and e in cols:
            rows = [
                {"gn": a, "es": b, "source": "agustin"}
                for a, b in zip(df[cols[g]], df[cols[e]])
                if isinstance(a, str) and isinstance(b, str)
            ]
            return rows, None
    return [], f"columnas no reconocidas: {list(df.columns)}"


def iter_kuatia_docs() -> Iterator[tuple[str, str]]:
    d = data.data_dir() / "kuatia"
    for subset in KUATIA_SUBSETS:
        p = data._require(d / f"{subset}_full.jsonl", "kuatia")
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if KUATIA_TEXT_FIELD not in rec:
                    raise ValueError(f"{p}: sin campo '{KUATIA_TEXT_FIELD}'; campos: {sorted(rec)}")
                yield subset, rec[KUATIA_TEXT_FIELD]


def iter_wikipedia_docs() -> Iterator[tuple[str, str]]:
    import pandas as pd

    p = data._require(data.data_dir() / "wikipedia" / "gn.parquet", "wikipedia")
    for text in pd.read_parquet(p, columns=["text"])["text"]:
        if isinstance(text, str) and text.strip():
            yield "wikipedia", text


def allowed_sources(path: Path) -> set[str]:
    # Solo la primera tabla del fichero: se corta en la primera línea que no es fila de tabla.
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif rows:
            break
    header = [h.lower() for h in rows[0]] if rows else []
    col = next((c for c in ("commercial", "comercial") if c in header), None)
    if "source" not in header or col is None:
        raise ValueError(f"{path}: falta una tabla con columnas 'source' y 'commercial'")
    si, ci = header.index("source"), header.index(col)
    allowed = set()
    for r in rows[1:]:
        if len(r) <= max(si, ci) or set(r[si]) <= set("-: "):
            continue
        if r[ci].lower() in ("yes", "sí", "si"):
            allowed |= {s.strip().strip("`") for s in r[si].split(",") if s.strip()}
    return allowed


def prepare() -> dict:
    keys = eval_keys()
    pairs = [{"gn": r["gn"], "es": r["es"], "source": "jojajovai"} for r in data.load_jojajovai("train")]
    pairs += load_americasnlp()
    agustin, note = load_agustin()
    pairs += agustin
    kept, stats = clean_pairs(pairs, keys)
    leaked = [p for p in kept if decont_key(p["gn"]) in keys or decont_key(p["es"]) in keys]
    if leaked:
        raise RuntimeError(f"{len(leaked)} frases de evaluación siguen en el entrenamiento tras decontaminar")
    stats |= {"out": len(kept), "by_source": dict(Counter(p["source"] for p in kept)), "agustin": note or "ok"}
    write_jsonl(train_dir() / "parallel.jsonl", kept)
    (train_dir() / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
