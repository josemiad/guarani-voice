"""Carga de conjuntos de evaluación desde data/ (o $GUANANI_DATA)."""

import csv
import json
import os
from pathlib import Path

HOW = {
    "jojajovai": "Ejecuta scripts/download.sh",
    "flores": (
        "Acepta las condiciones en https://huggingface.co/datasets/openlanguagedata/flores_plus, "
        "ejecuta `uv run hf auth login` y luego scripts/download.sh"
    ),
    "common_voice": (
        "Ejecuta `uv run hf auth login` y luego scripts/download.sh "
        "(descarga Common Voice 17 gn desde la réplica fsicoli/common_voice_17_0)"
    ),
    "americasnlp": "Ejecuta scripts/download.sh (AmericasNLP 2021, guarani-spanish)",
    "kuatia": (
        "Acepta las condiciones en https://huggingface.co/datasets/guaran-ia/kuatia "
        "y ejecuta scripts/download.sh"
    ),
    "train": "Ejecuta `guanani train prepare`",
    "bt": "Ejecuta `guanani train backtranslate --name <btN>` con el nombre que falta (p. ej. bt2)",
    "wikipedia": "Ejecuta scripts/download.sh (Wikipedia en guaraní, wikimedia/wikipedia 20231101.gn)",
}


class DataMissing(FileNotFoundError):
    pass


def data_dir() -> Path:
    return Path(os.environ.get("GUANANI_DATA", "data"))


def _require(path: Path, key: str) -> Path:
    if not path.exists():
        raise DataMissing(f"Falta {path}. {HOW[key]}")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_flores(split: str = "devtest") -> list[dict]:
    d = data_dir() / "flores"
    gn = _read_jsonl(_require(d / f"{split}.gug_Latn.jsonl", "flores"))
    es = {r["id"]: r["text"] for r in _read_jsonl(_require(d / f"{split}.spa_Latn.jsonl", "flores"))}
    return [{"gn": r["text"], "es": es[r["id"]]} for r in gn]


def load_jojajovai(split: str = "test") -> list[dict]:
    p = _require(data_dir() / "jojajovai" / "jojajovai_all.csv", "jojajovai")
    with p.open(encoding="utf-8", newline="") as f:
        return [{"gn": r["gn"], "es": r["es"], "source": r["source"]} for r in csv.DictReader(f) if r["split"] == split]


def load_jojajovai_annotated() -> list[dict]:
    p = _require(data_dir() / "jojajovai" / "jojajovai_sample_annotations.csv", "jojajovai")
    with p.open(encoding="utf-8", newline="") as f:
        return [
            {"gn": r["gn"], "es": r["es"], "source": r["source"], "dialect": r["dialect_unified"]}
            for r in csv.DictReader(f)
        ]


def load_common_voice(split: str = "test") -> list[dict]:
    d = data_dir() / "common_voice"
    p = _require(d / f"{split}.tsv", "common_voice")
    with p.open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [{"audio": str(d / "clips" / r["path"]), "gn": r["sentence"], "speaker": r.get("client_id", "")}
                for r in rows]


DATASETS = {
    "flores": load_flores,
    "jojajovai": load_jojajovai,
    "jojajovai-anotado": load_jojajovai_annotated,
    "cv": load_common_voice,
    "cv-dev": lambda: load_common_voice("dev"),
}
