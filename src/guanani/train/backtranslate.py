"""Retrotraducción gn→es a data/train/<name>/, por bloques y reanudable."""

import json
import re
import shutil
from collections import Counter
from pathlib import Path

from guanani import data, nmt
from guanani.train.corpus import clean_sentences, reservoir
from guanani.train.sources import (
    eval_keys,
    iter_kuatia_docs,
    iter_wikipedia_docs,
    read_jsonl,
    train_dir,
    write_jsonl,
)

BT_MODEL = "facebook/nllb-200-distilled-1.3B"


def select_sentences(n: int, seed: int = 42, lid: bool = True) -> tuple[list[dict], dict]:
    keys = eval_keys()
    stats, seen = Counter(), set()
    reject, threshold = None, None
    if lid:
        from guanani.train.lid import GlotLID, calibrate_threshold, load_glotlid

        model = load_glotlid()
        real = read_jsonl(data._require(train_dir() / "parallel.jsonl", "train"))
        threshold = calibrate_threshold([GlotLID(model).accepted_prob(p["gn"]) for p in real])
        reject = GlotLID(model, threshold).reject_label
    # Wikipedia entera (guaraní formal y técnico); KuatIA rellena hasta n. `seen` compartido: sin duplicados entre ambas.
    wiki = list(clean_sentences(iter_wikipedia_docs(), keys, stats, seen, reject))[:n]
    rest = reservoir(clean_sentences(iter_kuatia_docs(), keys, stats, seen, reject), n - len(wiki), seed)
    rows = [{"gn": s, "source": src} for src, s in wiki + rest]
    return rows, {"n": len(rows), "lid_threshold": threshold, "counts": dict(sorted(stats.items()))}


def backtranslate(
    name: str,
    n: int = 500000,
    model: str = BT_MODEL,
    sample: bool = False,
    batch_size: int = 16,
    chunk: int = 1000,
    seed: int = 42,
    lid: bool = True,
    from_: str | None = None,
) -> Path:
    # Solo btN: `bt` es el fichero de E2 (data/train/bt.jsonl) y otros nombres no se podrían cargar al entrenar.
    if not re.fullmatch(r"bt\d+", name):
        raise ValueError(f"--name debe ser btN (p. ej. bt2), no {name!r}")
    d = train_dir() / name
    meta = {"n": n, "model": model, "sample": sample, "chunk": chunk, "seed": seed, "lid": lid, "from": from_}
    meta_path = d / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != meta:
            raise ValueError(f"{d} se generó con {old} y ahora se pide {meta}: usa otro --name o borra {d}")
    else:
        d.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    src_path = d / "source.jsonl"
    if not src_path.exists():
        tmp = d / "source.jsonl.tmp"
        if from_:
            origin = data._require(train_dir() / from_ / "source.jsonl", "bt")
            if (origin.parent / "stats.json").exists():
                shutil.copyfile(origin.parent / "stats.json", d / "stats.json")
            shutil.copyfile(origin, tmp)
        else:
            rows, stats = select_sentences(n, seed, lid)
            (d / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            write_jsonl(tmp, rows)
        tmp.rename(src_path)
    rows = read_jsonl(src_path)

    parts = d / "parts"
    parts.mkdir(exist_ok=True)
    for i in range(0, len(rows), chunk):
        part = parts / f"{i // chunk:05d}.jsonl"
        if part.exists():
            continue
        block = rows[i : i + chunk]
        es = nmt.translate([r["gn"] for r in block], "gn", "es", model=model, batch_size=batch_size, sample=sample)
        tmp = part.with_suffix(".jsonl.tmp")
        write_jsonl(tmp, [{"gn": r["gn"], "es": e, "source": f"bt:{r['source']}"} for r, e in zip(block, es)])
        tmp.rename(part)

    out = d / f"{name}.jsonl"
    write_jsonl(out, [r for p in sorted(parts.glob("*.jsonl")) for r in read_jsonl(p)])
    return out
