# Subproject 1 — LoRA NLLB-1.3B gn↔es: Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare a clean, decontaminated gn–es parallel corpus, generate back-translation from KuatIA, train LoRA on NLLB-1.3B on the 3060 (E1 without and E2 with back-translation), and measure whether the thresholds are met on FLORES+ devtest.

**Architecture:** A `guanani.train` subpackage:
- `corpus.py`: pure functions for cleaning, decontamination, sampling and examples.
- `sources.py`: corpus reading and `prepare()`.
- `backtranslate.py`: resumable block-wise back-translation.
- `lora.py`: tokenisation, a dev chrF++ callback, `Seq2SeqTrainer` + `peft`, adapter merging.

The CLI subcommand is `guanani train prepare|backtranslate|lora`. The merged model is evaluated with the existing `guanani eval`, unchanged.

**Tech Stack:** Python 3.12, uv, torch, transformers 4.57 (`Seq2SeqTrainer`), peft, accelerate, pandas + pyarrow (parquet), sacrebleu, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-subproject-1-nmt-lora-design.md`

## Global Constraints

- Base: `facebook/nllb-200-distilled-1.3B`; codes `grn_Latn` / `spa_Latn`.
- LoRA: r=16, alpha=32, dropout 0.05; `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`.
- Defaults: AdamW lr 2e-4, cosine, 3 % warmup, batch 8 × accumulation 4, 2 epochs, max length 200 tokens, gradient checkpointing, seed 42, bf16.
- Selection: every 1,000 steps, greedy chrF++ on the first 300 FLORES+ dev sentences, averaged over both directions.
- Filters:
  - empty sides, identical sides, > 200 words, word ratio > 3, duplicates;
  - decontamination against FLORES+ dev+devtest, Jojajovai test + annotated, and Common Voice test.
- Back-translation:
  - 60,000 sentences of 5–60 words from `madlad400`, `culturax`, `fineweb-2`, `hplt-3`, `FinePDF`, `guarania-scraped`, `multi-wiki-qa`;
  - blocks of 1,000;
  - es→gn examples only.
- Thresholds (FLORES+ devtest): es→gn ≥ 36.05; gn→es ≥ 39.85; `jojajovai-anotado` gn→es ≥ 49.31.
- `data/` and `models/` are not versioned. Leave the user's GPU services (`llama-server`, `server.py`) alone; ~7 GB of VRAM is free.
- When killing processes, use patterns that do not match the command itself (`pkill -f "[r]un_e1"`).
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A KuatIA record without the expected text field → a clear error listing the fields, not a `KeyError` (Task 2).
2. An evaluation sentence in training data with different punctuation, case or apostrophe → removed (Task 1).
3. Back-translation interrupted mid-block → no duplicated or lost lines on resume (Task 3).
4. `--data parallel,bt` without `bt.jsonl` → `DataMissing` with instructions, before loading any model (Task 5).
5. Training resumed with `--resume` → neither the best checkpoint nor its dev score is lost (Task 4).

---

### Task 1: Pure corpus functions

**Files:**
- Create: `src/guanani/train/__init__.py` (empty), `src/guanani/train/corpus.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `guanani.normalize.normalize`
- Produces:
  - `KUATIA_SUBSETS: tuple[str, ...]`
  - `norm_es(text: str) -> str`
  - `decont_key(text: str) -> str`
  - `clean_pairs(pairs: Iterable[dict], eval_keys: set[str]) -> tuple[list[dict], dict]` — input `{gn, es, source}`; outputs normalised pairs and `stats` with keys `in`, `empty`, `identical`, `too_long`, `ratio`, `duplicate`, `contaminated`
  - `split_sentences(text: str) -> list[str]`
  - `sample_sentences(docs: Iterable[tuple[str, str]], n: int, eval_keys: set[str], seed: int = 42, min_words: int = 5, max_words: int = 60) -> list[tuple[str, str]]` — `docs` are `(subset, text)`; returns `(subset, sentence)`
  - `build_examples(parallel: list[dict], bt: list[dict] = ()) -> list[dict]` — examples `{src, tgt, src_lang, tgt_lang}`

- [ ] **Step 1: Write the failing tests**

`tests/test_corpus.py`:

```python
from guanani.train.corpus import (
    KUATIA_SUBSETS,
    build_examples,
    clean_pairs,
    decont_key,
    sample_sentences,
    split_sentences,
)


def _p(gn, es, source="s"):
    return {"gn": gn, "es": es, "source": source}


def test_clean_pairs_filters_each_reason():
    pairs = [
        _p("", "algo"),
        _p("hola", "Hola"),
        _p(" ".join(["ñe'ẽ"] * 201), " ".join(["palabra"] * 201)),
        _p("peteĩ", "uno dos tres cuatro"),
        _p("che róga", "mi casa"),
        _p("Che  róga", "mi casa"),
        _p("ko ñe'ẽ", "esta frase"),
    ]
    kept, stats = clean_pairs(pairs, eval_keys={decont_key("esta frase")})
    assert kept == [{"gn": "che róga", "es": "mi casa", "source": "s"}]
    assert stats == {
        "in": 7, "empty": 1, "identical": 1, "too_long": 1, "ratio": 1, "duplicate": 1, "contaminated": 1,
    }


def test_decont_key_ignores_case_punctuation_and_apostrophe_variants():
    assert decont_key("¡Mba’éichapa, REIKO!") == decont_key("mba'éichapa reiko")


def test_contaminated_gn_side_removed_even_with_variant_spelling():
    keys = {decont_key("Omopotĩ hikuái.")}
    kept, stats = clean_pairs([_p("omopotî hikuái", "limpiaron")], keys)
    assert kept == [] and stats["contaminated"] == 1


def test_split_sentences():
    assert split_sentences("Peteĩ. Mokõi! Mbohapy? Irundy") == ["Peteĩ.", "Mokõi!", "Mbohapy?", "Irundy"]


def test_sample_sentences_filters_dedups_decontaminates_and_is_deterministic():
    docs = [
        ("madlad400", "Ko ñe'ẽ ipuku porã ha iporã. Mbykymi. Ko ñe'ẽ ipuku porã ha iporã."),
        ("hplt-3", "Ñande ñe'ẽ oñeñe'ẽ Paraguái retãme. Ambue ñe'ẽ ipuku porã avei ko'ápe."),
    ]
    keys = {decont_key("Ambue ñe'ẽ ipuku porã avei ko'ápe.")}
    got = sample_sentences(docs, n=10, eval_keys=keys, seed=1)
    assert sorted(s for _, s in got) == ["Ko ñe'ẽ ipuku porã ha iporã.", "Ñande ñe'ẽ oñeñe'ẽ Paraguái retãme."]
    assert sample_sentences(docs, n=1, eval_keys=keys, seed=1) == sample_sentences(docs, n=1, eval_keys=keys, seed=1)


def test_kuatia_whitelist_excludes_eval_and_informal_sources():
    for bad in ("flores-200", "belele", "jojajovai", "commonvoice", "americasnlp2024", "bible", "twitter_covid_py"):
        assert bad not in KUATIA_SUBSETS


def test_build_examples_real_pairs_both_directions_bt_only_es_gn():
    ex = build_examples([_p("che", "yo")], bt=[_p("ore", "nosotros", "bt:madlad400")])
    assert ex == [
        {"src": "che", "tgt": "yo", "src_lang": "grn_Latn", "tgt_lang": "spa_Latn"},
        {"src": "yo", "tgt": "che", "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
        {"src": "nosotros", "tgt": "ore", "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
    ]
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_corpus.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'guanani.train'`

- [ ] **Step 3: Implement `src/guanani/train/__init__.py` (empty) and `src/guanani/train/corpus.py`**

```python
"""Limpieza, deduplicación, decontaminación y muestreo de corpus gn–es (funciones puras)."""

import random
import re
import unicodedata
from collections.abc import Iterable

from guanani.normalize import normalize

MAX_WORDS = 200
MAX_RATIO = 3.0
# Subconjuntos de KuatIA de registro probablemente formal. Excluidos a propósito: fuentes de evaluación
# (flores-200, belele, jojajovai, commonvoice, americasnlp*, americasnli), Twitter, chat/afectivos y Biblia.
KUATIA_SUBSETS = ("madlad400", "culturax", "fineweb-2", "hplt-3", "FinePDF", "guarania-scraped", "multi-wiki-qa")

_SENT = re.compile(r"(?<=[.!?])\s+")


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
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def sample_sentences(
    docs: Iterable[tuple[str, str]],
    n: int,
    eval_keys: set[str],
    seed: int = 42,
    min_words: int = 5,
    max_words: int = 60,
) -> list[tuple[str, str]]:
    # Muestreo de reservorio: una pasada, memoria O(n) + claves vistas.
    rng = random.Random(seed)
    seen, sample, k = set(), [], 0
    for source, doc in docs:
        for s in split_sentences(normalize(doc)):
            if not min_words <= len(s.split()) <= max_words:
                continue
            key = decont_key(s)
            if key in seen or key in eval_keys:
                continue
            seen.add(key)
            k += 1
            if len(sample) < n:
                sample.append((source, s))
            else:
                j = rng.randrange(k)
                if j < n:
                    sample[j] = (source, s)
    return sample


def build_examples(parallel: list[dict], bt: list[dict] = ()) -> list[dict]:
    ex = []
    for p in parallel:
        ex.append({"src": p["gn"], "tgt": p["es"], "src_lang": "grn_Latn", "tgt_lang": "spa_Latn"})
        ex.append({"src": p["es"], "tgt": p["gn"], "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"})
    # Retrotraducción solo es→gn: el español es sintético, el guaraní es real.
    for p in bt:
        ex.append({"src": p["es"], "tgt": p["gn"], "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"})
    return ex
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_corpus.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/guanani/train/__init__.py src/guanani/train/corpus.py tests/test_corpus.py
git commit -m "feat: funciones de limpieza, decontaminación y muestreo de corpus

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Data sources, `prepare()` and download

**Files:**
- Modify: `pyproject.toml` (dependencies), `src/guanani/data.py` (keys in `HOW`), `scripts/download.sh`, `.gitignore`
- Create: `src/guanani/train/sources.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: `data.data_dir`, `data._require`, `data.HOW`, `data.load_flores`, `data.load_jojajovai`, `data.load_jojajovai_annotated`, `data.load_common_voice`, `corpus.*`
- Produces:
  - `train_dir() -> Path` (`data_dir() / "train"`)
  - `read_jsonl(path: Path) -> list[dict]`, `write_jsonl(path: Path, rows: Iterable[dict]) -> None`
  - `eval_keys() -> set[str]`
  - `load_americasnlp() -> list[dict]`
  - `load_agustin() -> tuple[list[dict], str | None]` (a note with the reason if skipped)
  - `KUATIA_TEXT_FIELD = "text"`; `iter_kuatia_docs() -> Iterator[tuple[str, str]]`
  - `prepare() -> dict` (writes `parallel.jsonl` and `stats.json`)

On-disk format:
- `data/americasnlp/train.gn`, `data/americasnlp/train.es` (aligned lines)
- `data/agustin/train.parquet` (optional)
- `data/kuatia/<subset>_full.jsonl` (one JSON line per document)

- [ ] **Step 1: Add dependencies and ignore `models/`**

In `pyproject.toml`, inside `dependencies`, add:

```toml
  "peft>=0.17",
  "accelerate>=1.0",
  "pyarrow",
```

Add the line `models/` to `.gitignore`.

Run: `UV_HTTP_TIMEOUT=300 uv sync -q && uv run python -c "import peft, accelerate, pyarrow; print(peft.__version__)"`
Expected: prints the peft version without error.

- [ ] **Step 2: Write the failing tests**

`tests/test_sources.py`:

```python
import json

import pytest

from guanani import data
from guanani.train import sources


@pytest.fixture
def ddir(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    return tmp_path


def _write_eval_sets(d):
    (d / "flores").mkdir()
    for split in ("dev", "devtest"):
        for code, text in (("gug_Latn", f"Ñe'ẽ {split}"), ("spa_Latn", f"Frase {split}")):
            (d / "flores" / f"{split}.{code}.jsonl").write_text(json.dumps({"id": 1, "text": text}), encoding="utf-8")
    (d / "jojajovai").mkdir()
    (d / "jojajovai" / "jojajovai_all.csv").write_text(
        "split,source,gn,es,tokens_gn,tokens_es\n"
        "train,abc,che róga,mi casa,x,y\n"
        "train,abc,Ñe'ẽ devtest,otra cosa distinta,x,y\n"
        "test,abc,ko mba'e,esta cosa,x,y\n",
        encoding="utf-8",
    )
    (d / "jojajovai" / "jojajovai_sample_annotations.csv").write_text(
        "source,gn,es,dialect_unified\nblogs,ko mba'e,esta cosa,guarani\n", encoding="utf-8"
    )
    (d / "common_voice").mkdir()
    (d / "common_voice" / "test.tsv").write_text("path\tsentence\na.mp3\tMba'éichapa reiko\n", encoding="utf-8")


def test_prepare_writes_parallel_and_stats_without_eval_sentences(ddir):
    _write_eval_sets(ddir)
    (ddir / "americasnlp").mkdir()
    (ddir / "americasnlp" / "train.gn").write_text("che róga\nmba'éichapa reiko!\n", encoding="utf-8")
    (ddir / "americasnlp" / "train.es").write_text("mi casa\n¿cómo estás?\n", encoding="utf-8")
    stats = sources.prepare()
    rows = sources.read_jsonl(ddir / "train" / "parallel.jsonl")
    assert rows == [{"gn": "che róga", "es": "mi casa", "source": "jojajovai"}]
    assert stats["contaminated"] == 2 and stats["duplicate"] == 1
    assert stats["agustin"] == "no descargado"
    assert json.loads((ddir / "train" / "stats.json").read_text(encoding="utf-8"))["out"] == 1


def test_americasnlp_misaligned_raises(ddir):
    (ddir / "americasnlp").mkdir()
    (ddir / "americasnlp" / "train.gn").write_text("a\nb\n", encoding="utf-8")
    (ddir / "americasnlp" / "train.es").write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="alineados"):
        sources.load_americasnlp()


def test_kuatia_missing_subset_explains_how(ddir):
    with pytest.raises(data.DataMissing, match="kuatia"):
        next(sources.iter_kuatia_docs())


def test_kuatia_record_without_text_field_lists_fields(ddir):
    d = ddir / "kuatia"
    d.mkdir()
    (d / "madlad400_full.jsonl").write_text(json.dumps({"content": "x", "lang": "gn"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content"):
        next(sources.iter_kuatia_docs())


def test_agustin_unknown_columns_is_skipped_with_note(ddir):
    import pandas as pd

    (ddir / "agustin").mkdir()
    pd.DataFrame({"prompt": ["a"], "answer": ["b"]}).to_parquet(ddir / "agustin" / "train.parquet")
    rows, note = sources.load_agustin()
    assert rows == [] and "prompt" in note
```

- [ ] **Step 3: Run and verify it fails**

Run: `uv run pytest tests/test_sources.py -q`
Expected: FAIL with `ImportError: cannot import name 'sources'`

- [ ] **Step 4: Add keys to `data.HOW` in `src/guanani/data.py`**

Inside the `HOW` dictionary, add:

```python
    "americasnlp": "Ejecuta scripts/download.sh (AmericasNLP 2021, guarani-spanish)",
    "kuatia": (
        "Acepta las condiciones en https://huggingface.co/datasets/guaran-ia/kuatia "
        "y ejecuta scripts/download.sh"
    ),
    "train": "Ejecuta `guanani train prepare`",
    "bt": "Ejecuta `guanani train backtranslate`",
```

- [ ] **Step 5: Implement `src/guanani/train/sources.py`**

```python
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
```

- [ ] **Step 6: Run and verify it passes**

Run: `uv run pytest tests/test_sources.py -q`
Expected: 5 passed. In `test_prepare_…`: `che róga`/`mi casa` appears in both Jojajovai and AmericasNLP → 1 duplicate; `Ñe'ẽ devtest` and `mba'éichapa reiko!` are evaluation sentences → 2 contaminated.

- [ ] **Step 7: Extend `scripts/download.sh`**

Before the FLORES+ section, add:

```bash
mkdir -p data/americasnlp data/kuatia data/agustin
anlp=https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/guarani-spanish
for f in train.gn train.es; do
  curl -fsSL -o "data/americasnlp/$f" "$anlp/$f"
done
echo "AmericasNLP OK"
```

At the end of the script, add:

```bash
uv run python - <<'EOF' || echo "KuatIA falló: acepta las condiciones en https://huggingface.co/datasets/guaran-ia/kuatia"
import shutil
from huggingface_hub import hf_hub_download
from guanani.train.corpus import KUATIA_SUBSETS
for s in KUATIA_SUBSETS:
    p = hf_hub_download("guaran-ia/kuatia", f"individual_full/{s}_full.jsonl", repo_type="dataset")
    shutil.copy(p, f"data/kuatia/{s}_full.jsonl")
print("KuatIA OK")
EOF

uv run python - <<'EOF' || echo "agustin (opcional) falló: acepta las condiciones en https://huggingface.co/datasets/guaran-ia/agustin-guarani-llm-splits"
import shutil
from huggingface_hub import hf_hub_download
shutil.copy(hf_hub_download("guaran-ia/agustin-guarani-llm-splits", "train.parquet", repo_type="dataset"), "data/agustin/train.parquet")
print("agustin OK")
EOF
```

Run: `bash -n scripts/download.sh && echo ok`
Expected: `ok`

- [ ] **Step 8: Download and check the real schema**

Run: `UV_HTTP_TIMEOUT=300 ./scripts/download.sh 2>&1 | grep -E "OK|falló"; wc -l data/americasnlp/*; head -c 400 data/kuatia/madlad400_full.jsonl; echo; uv run python -c "import pandas as pd; df=pd.read_parquet('data/agustin/train.parquet'); print(df.columns.tolist(), len(df))"`

Expected:
- Jojajovai, AmericasNLP, FLORES+ and Common Voice report OK.
- `train.gn`/`train.es` have 26032 lines.
- The first KuatIA line shows its fields.

If something doesn't match:
- **KuatIA or agustin fails on access:** ask the user to accept the conditions, and continue with tasks 3–5, which don't need real data.
- **KuatIA's text field is not called `text`:** change `KUATIA_TEXT_FIELD` in `sources.py`. If the real name happens to be `content`, also adjust the `content` field in `test_kuatia_record_without_text_field_lists_fields`. Record a `Ruling:`.
- **agustin's columns are not in `AGUSTIN_COLUMNS`:** if there is an obvious gn/es pair under another name, add it to `AGUSTIN_COLUMNS` and record a `Ruling:`. If there is no clear pair, leave agustin skipped.

- [ ] **Step 9: Run `prepare()` on real data**

Run: `uv run python -c "from guanani.train.sources import prepare; import json; print(json.dumps(prepare(), indent=1))"`
Expected: `in` ≈ 46k (without agustin), `out` between 30k and 46k, `contaminated` > 0, and `by_source` with `jojajovai` and `americasnlp`.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/guanani/data.py src/guanani/train/sources.py scripts/download.sh tests/test_sources.py
git commit -m "feat: fuentes de entrenamiento, decontaminación y prepare()

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Resumable back-translation

**Files:**
- Create: `src/guanani/train/backtranslate.py`
- Test: `tests/test_backtranslate.py`

**Interfaces:**
- Consumes: `sources.train_dir`, `sources.read_jsonl`, `sources.write_jsonl`, `sources.eval_keys`, `sources.iter_kuatia_docs`, `corpus.sample_sentences`, `guanani.nmt.translate` (accessed as `nmt.translate` so tests can replace it)
- Produces: `BT_MODEL = "facebook/nllb-200-distilled-1.3B"`; `backtranslate(n: int = 60000, model: str = BT_MODEL, chunk: int = 1000, seed: int = 42) -> Path` (path to `bt.jsonl`)

Files:
- `data/train/bt_source.jsonl`: `{gn, source}`, the fixed sample.
- `data/train/bt_parts/<NNNNN>.jsonl`: one file per block, written via `.tmp` + rename.
- `data/train/bt.jsonl`: the final concatenation.

- [ ] **Step 1: Write the failing tests**

`tests/test_backtranslate.py`:

```python
import pytest

from guanani import nmt
from guanani.train import backtranslate as bt
from guanani.train import sources


@pytest.fixture
def ddir(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sents = [{"gn": f"frase {i}", "source": "madlad400"} for i in range(25)]
    sources.write_jsonl(tmp_path / "train" / "bt_source.jsonl", sents)
    calls = []

    def fake_translate(texts, src, tgt, model=None, **kw):
        assert (src, tgt) == ("gn", "es")
        calls.append(len(texts))
        return [t.upper() for t in texts]

    monkeypatch.setattr(nmt, "translate", fake_translate)
    return tmp_path, calls


def test_backtranslate_writes_all_rows_in_order(ddir):
    tmp, calls = ddir
    out = bt.backtranslate(n=25, chunk=10)
    rows = sources.read_jsonl(out)
    assert [r["gn"] for r in rows] == [f"frase {i}" for i in range(25)]
    assert rows[3] == {"gn": "frase 3", "es": "FRASE 3", "source": "bt:madlad400"}
    assert calls == [10, 10, 5]


def test_backtranslate_resumes_without_duplicates(ddir):
    tmp, calls = ddir
    bt.backtranslate(n=25, chunk=10)
    parts = tmp / "train" / "bt_parts"
    (parts / "00001.jsonl").unlink()
    (parts / "00002.jsonl").unlink()
    # Simula un corte a mitad del bloque 2: solo quedó el .tmp a medio escribir.
    (parts / "00002.jsonl.tmp").write_text("{\"parcial", encoding="utf-8")
    calls.clear()
    rows = sources.read_jsonl(bt.backtranslate(n=25, chunk=10))
    assert calls == [10, 5]
    assert [r["gn"] for r in rows] == [f"frase {i}" for i in range(25)]
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_backtranslate.py -q`
Expected: FAIL with `ImportError: cannot import name 'backtranslate'`

- [ ] **Step 3: Implement `src/guanani/train/backtranslate.py`**

```python
"""Retrotraducción gn→es de frases de KuatIA, por bloques y reanudable."""

from pathlib import Path

from guanani import nmt
from guanani.train.corpus import sample_sentences
from guanani.train.sources import eval_keys, iter_kuatia_docs, read_jsonl, train_dir, write_jsonl

BT_MODEL = "facebook/nllb-200-distilled-1.3B"


def backtranslate(n: int = 60000, model: str = BT_MODEL, chunk: int = 1000, seed: int = 42) -> Path:
    src_path = train_dir() / "bt_source.jsonl"
    if not src_path.exists():
        sample = sample_sentences(iter_kuatia_docs(), n, eval_keys(), seed=seed)
        write_jsonl(src_path, [{"gn": s, "source": source} for source, s in sample])
    rows = read_jsonl(src_path)

    parts = train_dir() / "bt_parts"
    parts.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), chunk):
        part = parts / f"{i // chunk:05d}.jsonl"
        if part.exists():
            continue
        block = rows[i : i + chunk]
        es = nmt.translate([r["gn"] for r in block], "gn", "es", model=model)
        tmp = part.with_suffix(".jsonl.tmp")
        write_jsonl(tmp, [{"gn": r["gn"], "es": e, "source": f"bt:{r['source']}"} for r, e in zip(block, es)])
        tmp.rename(part)

    out = train_dir() / "bt.jsonl"
    write_jsonl(out, [r for p in sorted(parts.glob("*.jsonl")) for r in read_jsonl(p)])
    return out
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_backtranslate.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/guanani/train/backtranslate.py tests/test_backtranslate.py
git commit -m "feat: retrotraducción de KuatIA por bloques reanudable

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Launch the real back-translation in the background** (only if KuatIA is downloaded)

Run:
```bash
nohup env UV_HTTP_TIMEOUT=300 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -c "from guanani.train.backtranslate import backtranslate; print(backtranslate())" \
  > /tmp/guanani-bt.log 2>&1 &
```
Expected: `data/train/bt_source.jsonl` with 60,000 lines within a few minutes, and `bt_parts/*.jsonl` files appearing at ~1 every 1–3 minutes.

Don't wait for it: continue with Task 4. This task occupies the GPU, so the Task 6 memory test must wait until it finishes. If KuatIA is not available, note it and continue; E2 stays blocked until it is.

---

### Task 4: LoRA training and merge

**Files:**
- Create: `src/guanani/train/lora.py`
- Test: `tests/test_lora.py`, modify `tests/test_models_slow.py`

**Interfaces:**
- Consumes: `corpus.build_examples`, `sources.train_dir`, `sources.read_jsonl`, `data._require`, `data.load_flores`, `eval.score_mt`, `normalize`
- Produces:
  - `BASE_MODEL`, `TARGET_MODULES`, `MAX_LEN = 200`
  - `tokenize_examples(tok, examples: list[dict], max_len: int = MAX_LEN) -> list[dict]` (`input_ids`, `attention_mask`, `labels`)
  - `dev_chrf(model, tok, pairs: list[dict], direction: str, batch_size: int = 16) -> float`
  - `class DevChrF(TrainerCallback)`:
    - constructor `__init__(tok, dev_pairs, eval_steps, out_dir: Path)`;
    - attributes `best: float`, `best_step: int | None`, `history: dict[str, dict]`;
    - persists to `out_dir / "dev_history.json"` and saves the best adapter to `out_dir / "best_adapter"`.
  - `train(name, examples, dev_pairs, base=BASE_MODEL, epochs=2.0, batch_size=8, grad_accum=4, lr=2e-4, max_steps=-1, eval_steps=1000, resume=False, models_dir=Path("models")) -> dict` (returns `{"best_step", "best_dev", "history"}`)
  - `merge(base: str, adapter_dir: Path, out_dir: Path) -> None`
  - `run(name: str, data_kinds: list[str], dev_size: int = 300, models_dir: Path = Path("models"), **train_kwargs) -> Path`, which:
    1. validates `data_kinds ⊆ {"parallel", "bt"}` (otherwise `ValueError`);
    2. checks the files with `data._require` before loading any model;
    3. trains, merges and writes `train_meta.json`;
    4. returns `models_dir / name`.

- [ ] **Step 1: Write the failing fast tests**

`tests/test_lora.py`:

```python
import json

import pytest

from guanani import data
from guanani.train import lora


def test_tokenize_examples_sets_language_codes():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    ex = [
        {"src": "che", "tgt": "yo", "src_lang": "grn_Latn", "tgt_lang": "spa_Latn"},
        {"src": "yo", "tgt": "che", "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
    ]
    out = lora.tokenize_examples(tok, ex)
    firsts = {(tok.convert_ids_to_tokens(o["input_ids"][0]), tok.convert_ids_to_tokens(o["labels"][0])) for o in out}
    assert firsts == {("grn_Latn", "spa_Latn"), ("spa_Latn", "grn_Latn")}


def test_dev_history_survives_resume(tmp_path):
    (tmp_path / "dev_history.json").write_text(
        json.dumps({"1000": {"gn-es": 40.0, "es-gn": 34.0, "mean": 37.0}, "2000": {"gn-es": 41.0, "es-gn": 35.0, "mean": 38.0}}),
        encoding="utf-8",
    )
    cb = lora.DevChrF(tok=None, dev_pairs=[], eval_steps=1000, out_dir=tmp_path)
    assert cb.best == 38.0 and cb.best_step == 2000


def test_run_rejects_unknown_data_kind(tmp_path):
    with pytest.raises(ValueError, match="foo"):
        lora.run("x", ["parallel", "foo"], models_dir=tmp_path)


def test_run_missing_bt_raises_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "parallel.jsonl").write_text('{"gn": "a", "es": "b", "source": "s"}\n', encoding="utf-8")
    with pytest.raises(data.DataMissing, match="backtranslate"):
        lora.run("x", ["parallel", "bt"], models_dir=tmp_path)
```

- [ ] **Step 2: Add the slow test to `tests/test_models_slow.py`**

```python
def test_lora_tiny_training_merges_into_translatable_model(tmp_path):
    from guanani.nmt import translate
    from guanani.train.corpus import build_examples
    from guanani.train.lora import merge, train

    pairs = [{"gn": f"Che réra Juan {i}.", "es": f"Me llamo Juan {i}.", "source": "t"} for i in range(50)]
    dev = [{"gn": "Mba'éichapa reiko?", "es": "¿Cómo estás?"}] * 4
    base = "facebook/nllb-200-distilled-600M"
    res = train("tiny", build_examples(pairs), dev, base=base, batch_size=4, grad_accum=1,
                max_steps=20, eval_steps=10, models_dir=tmp_path)
    assert res["best_step"] in (10, 20)
    merge(base, tmp_path / "tiny" / "best_adapter", tmp_path / "tiny")
    out = translate(["Mba'éichapa reiko?"], "gn", "es", model=str(tmp_path / "tiny"))
    assert out[0].strip()
```

- [ ] **Step 3: Run and verify they fail**

Run: `uv run pytest tests/test_lora.py -q`
Expected: FAIL with `ImportError: cannot import name 'lora'`

- [ ] **Step 4: Implement `src/guanani/train/lora.py`**

```python
"""Fine-tuning LoRA de NLLB, selección por chrF++ en FLORES+ dev y fusión del adaptador."""

import gc
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from guanani import data
from guanani.eval import score_mt
from guanani.normalize import normalize
from guanani.train.corpus import build_examples
from guanani.train.sources import read_jsonl, train_dir

BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
MAX_LEN = 200
CODES = {"gn": "grn_Latn", "es": "spa_Latn"}
DATA_FILES = {"parallel": ("parallel.jsonl", "train"), "bt": ("bt.jsonl", "bt")}


def tokenize_examples(tok, examples: list[dict], max_len: int = MAX_LEN) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for ex in examples:
        groups.setdefault((ex["src_lang"], ex["tgt_lang"]), []).append(ex)
    out = []
    for (src_lang, tgt_lang), exs in groups.items():
        tok.src_lang, tok.tgt_lang = src_lang, tgt_lang
        enc = tok([e["src"] for e in exs], text_target=[e["tgt"] for e in exs], max_length=max_len, truncation=True)
        out += [
            {"input_ids": i, "attention_mask": a, "labels": lab}
            for i, a, lab in zip(enc["input_ids"], enc["attention_mask"], enc["labels"])
        ]
    return out


def dev_chrf(model, tok, pairs: list[dict], direction: str, batch_size: int = 16) -> float:
    src, tgt = direction.split("-")
    tok.src_lang = CODES[src]
    bos = tok.convert_tokens_to_ids(CODES[tgt])
    was_training = model.training
    model.eval()
    hyps = []
    for i in range(0, len(pairs), batch_size):
        enc = tok([p[src] for p in pairs[i : i + batch_size]], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**enc, forced_bos_token_id=bos, max_new_tokens=MAX_LEN, num_beams=1)
        hyps += tok.batch_decode(gen, skip_special_tokens=True)
    if was_training:
        model.train()
    refs = [p[tgt] for p in pairs]
    if tgt == "gn":
        hyps, refs = [normalize(h) for h in hyps], [normalize(r) for r in refs]
    return score_mt(hyps, refs)["chrf++"]


class DevChrF(TrainerCallback):
    def __init__(self, tok, dev_pairs: list[dict], eval_steps: int, out_dir: Path):
        self.tok, self.dev, self.eval_steps, self.out_dir = tok, dev_pairs, eval_steps, out_dir
        self.path = out_dir / "dev_history.json"
        self.history = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.best, self.best_step = float("-inf"), None
        for step, s in self.history.items():
            if s["mean"] > self.best:
                self.best, self.best_step = s["mean"], int(step)

    def _evaluate(self, model, step: int) -> None:
        scores = {d: dev_chrf(model, self.tok, self.dev, d) for d in ("gn-es", "es-gn")}
        scores["mean"] = round((scores["gn-es"] + scores["es-gn"]) / 2, 2)
        self.history[str(step)] = scores
        if scores["mean"] > self.best:
            self.best, self.best_step = scores["mean"], step
            model.save_pretrained(self.out_dir / "best_adapter")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        print(f"[dev] paso {step}: {scores} (mejor {self.best} en {self.best_step})", flush=True)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.eval_steps == 0:
            self._evaluate(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kw):
        if str(state.global_step) not in self.history:
            self._evaluate(model, state.global_step)


def train(
    name: str,
    examples: list[dict],
    dev_pairs: list[dict],
    base: str = BASE_MODEL,
    epochs: float = 2.0,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_steps: int = -1,
    eval_steps: int = 1000,
    resume: bool = False,
    models_dir: Path = Path("models"),
) -> dict:
    out = models_dir / name
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=TARGET_MODULES, task_type="SEQ_2_SEQ_LM"))
    model.print_trainable_parameters()
    cb = DevChrF(tok, dev_pairs, eval_steps, out)
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        max_steps=max_steps,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        report_to=[],
        seed=42,
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenize_examples(tok, examples),
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        processing_class=tok,
        callbacks=[cb],
    )
    oom = False
    try:
        trainer.train(resume_from_checkpoint=True if resume else None)
    except torch.OutOfMemoryError:
        oom = True
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    if oom:
        raise RuntimeError(
            f"Sin VRAM con --batch-size {batch_size}: prueba --batch-size {max(1, batch_size // 2)} "
            f"--grad-accum {grad_accum * 2} (mismo batch efectivo)"
        )
    return {"best_step": cb.best_step, "best_dev": cb.history.get(str(cb.best_step)), "history": cb.history}


def merge(base: str, adapter_dir: Path, out_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(base)
    m = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.float16)
    m = PeftModel.from_pretrained(m, str(adapter_dir)).merge_and_unload()
    m.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def run(name: str, data_kinds: list[str], dev_size: int = 300, models_dir: Path = Path("models"), **train_kwargs) -> Path:
    unknown = set(data_kinds) - set(DATA_FILES)
    if unknown:
        raise ValueError(f"--data desconocido: {sorted(unknown)}; opciones: {sorted(DATA_FILES)}")
    paths = {k: data._require(train_dir() / DATA_FILES[k][0], DATA_FILES[k][1]) for k in data_kinds}
    parallel = read_jsonl(paths["parallel"]) if "parallel" in paths else []
    bt = read_jsonl(paths["bt"]) if "bt" in paths else []
    examples = build_examples(parallel, bt)
    dev = data.load_flores("dev")[:dev_size]
    base = train_kwargs.get("base", BASE_MODEL)
    res = train(name, examples, dev, models_dir=models_dir, **train_kwargs)
    out = models_dir / name
    merge(base, out / "best_adapter", out)
    meta = {"name": name, "base": base, "data": data_kinds, "n_parallel": len(parallel), "n_bt": len(bt),
            "n_examples": len(examples), "dev_size": dev_size, "train_kwargs": train_kwargs} | res
    (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
```

- [ ] **Step 5: Run the fast tests and verify they pass**

Run: `uv run pytest tests/test_lora.py -q`
Expected: 4 passed

- [ ] **Step 6: Run the slow test** (only if the Task 3 back-translation is not using the GPU; if it is, wait until it finishes)

Run: `uv run pytest -m slow tests/test_models_slow.py::test_lora_tiny_training_merges_into_translatable_model -q`
Expected: 1 passed. If it fails because the transformers/peft API differs from what was expected (e.g. an argument name), apply the minimal change and record a `Ruling:`.

- [ ] **Step 7: Commit**

```bash
git add src/guanani/train/lora.py tests/test_lora.py tests/test_models_slow.py
git commit -m "feat: entrenamiento LoRA de NLLB con selección por chrF++ en dev y fusión

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The `guanani train` subcommand

**Files:**
- Modify: `src/guanani/cli.py`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `sources.prepare`, `backtranslate.backtranslate`, `lora.run`
- Produces:
  - `guanani train prepare`
  - `guanani train backtranslate [--n N] [--model M]`
  - `guanani train lora --name N [--data parallel|parallel,bt] [--epochs] [--batch-size] [--grad-accum] [--lr] [--max-steps] [--eval-steps] [--dev-size] [--base] [--resume]`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli.py`)

```python
def test_train_lora_missing_bt_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "parallel.jsonl").write_text('{"gn": "a", "es": "b", "source": "s"}\n', encoding="utf-8")
    assert main(["train", "lora", "--name", "x", "--data", "parallel,bt"]) == 2
    assert "backtranslate" in capsys.readouterr().err


def test_train_prepare_missing_data_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    assert main(["train", "prepare"]) == 2
    assert "Falta" in capsys.readouterr().err
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL (`argparse` exits with code 2 on `invalid choice: 'train'` → `SystemExit`)

- [ ] **Step 3: Modify `src/guanani/cli.py`**

Add `import json` with the other imports. After the `eval` parser (before `a = p.parse_args(argv)`), add:

```python
    tr = sub.add_parser("train", help="prepara datos y entrena LoRA de NLLB")
    tsub = tr.add_subparsers(dest="train_cmd", required=True)
    tsub.add_parser("prepare", help="construye data/train/parallel.jsonl")
    b = tsub.add_parser("backtranslate", help="retrotraduce frases de KuatIA a data/train/bt.jsonl")
    b.add_argument("--n", type=int, default=60000)
    b.add_argument("--model")
    lo = tsub.add_parser("lora", help="entrena LoRA, fusiona y guarda en models/<name>")
    lo.add_argument("--name", required=True)
    lo.add_argument("--data", default="parallel", help="parallel o parallel,bt")
    lo.add_argument("--epochs", type=float, default=2.0)
    lo.add_argument("--batch-size", type=int, default=8)
    lo.add_argument("--grad-accum", type=int, default=4)
    lo.add_argument("--lr", type=float, default=2e-4)
    lo.add_argument("--max-steps", type=int, default=-1)
    lo.add_argument("--eval-steps", type=int, default=1000)
    lo.add_argument("--dev-size", type=int, default=300)
    lo.add_argument("--base", default="facebook/nllb-200-distilled-1.3B")
    lo.add_argument("--resume", action="store_true")
```

In the `try` block, before the final `else:` for `eval`, add:

```python
        elif a.cmd == "train":
            if a.train_cmd == "prepare":
                from guanani.train.sources import prepare

                print(json.dumps(prepare(), ensure_ascii=False, indent=2))
            elif a.train_cmd == "backtranslate":
                from guanani.train.backtranslate import backtranslate

                print(backtranslate(n=a.n, **_kw(a)))
            else:
                from guanani.train.lora import run

                print(run(a.name, a.data.split(","), dev_size=a.dev_size, base=a.base, epochs=a.epochs,
                          batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr, max_steps=a.max_steps,
                          eval_steps=a.eval_steps, resume=a.resume))
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_cli.py -q && uv run pytest -q`
Expected: 4 passed in `test_cli.py`; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/guanani/cli.py tests/test_cli.py
git commit -m "feat: subcomando guanani train prepare|backtranslate|lora

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Memory test and experiment E1

**Files:**
- Create: `models/e1/` (not versioned), new rows in `results/`

- [ ] **Step 1: Memory and speed test** (once the back-translation has freed the GPU)

Run:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run guanani train lora --name probe --max-steps 50 --eval-steps 25 --dev-size 50 2>&1 | grep -E "it/s|s/it|\[dev\]|trainable|Error" | tail -5
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
rm -rf models/probe
```
Expected: finishes without OOM, and prints the trainable parameters (~1–2 %) and two `[dev]` lines.

Then:
- Compute s/step and estimate E1's duration: `steps = ceil(2 × n_parallel × 2 epochs / 32)`.
- If the estimate exceeds 10 h, use `--epochs 1` for E1 and E2 and record a `Ruling:`.
- On OOM, repeat with `--batch-size 4 --grad-accum 8` and record a `Ruling:`.

- [ ] **Step 2: Launch E1 in the background**

Run:
```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run guanani train lora --name e1 --data parallel \
  > models-e1.log 2>&1 &
```
Expected: `[dev] paso 1000: …` lines roughly every 1,000 steps in `models-e1.log`. If it is interrupted, relaunch the same command with `--resume`.

- [ ] **Step 3: Check the training result**

Run: `cat models/e1/train_meta.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['best_step'], d['best_dev'], d['n_examples'])"; ls models/e1`
Expected: `best_step` and `best_dev` present; `models/e1/` contains `config.json`, `model.safetensors`, `tokenizer.json`, `best_adapter/`.

- [ ] **Step 4: Evaluate E1**

Run:
```bash
for d in gn-es es-gn; do uv run guanani eval nmt flores --direction $d --model models/e1; done
uv run guanani eval nmt jojajovai-anotado --direction gn-es --model models/e1
for d in gn-es es-gn; do uv run guanani eval nmt jojajovai --direction $d --model models/e1; done
```
Expected: 5 new rows in `results/SUMMARY.md` with model `models/e1`.

- [ ] **Step 5: Commit the results**

```bash
git add results/
git commit -m "docs: resultados E1 (LoRA con paralelos reales)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Experiment E2 and conclusions

**Files:**
- Modify: `results/SUMMARY.md`
- Create: `models/e2/` (not versioned)

- [ ] **Step 1: Check the back-translation**

Run: `wc -l data/train/bt.jsonl; head -3 data/train/bt.jsonl`
Expected: ~60,000 lines with real `gn`, synthetic `es` and `source` `bt:<subset>`. If KuatIA was never available, E2 cannot be done: skip to Step 4 and say so in the conclusions.

- [ ] **Step 2: Launch E2**

Run:
```bash
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run guanani train lora --name e2 --data parallel,bt \
  > models-e2.log 2>&1 &
```
(Use the same `--epochs` as E1 if Task 6 recorded a `Ruling:`.)
Expected: `models/e2/train_meta.json` when it finishes.

- [ ] **Step 3: Evaluate E2** (the same 5 commands as Task 6 Step 4, with `--model models/e2`)

Expected: 5 new rows in `results/SUMMARY.md`.

- [ ] **Step 4: Write conclusions**

In `results/SUMMARY.md`, after the subproject 0 conclusions and before the table, insert a section `## Conclusions (subproject 1, <date>)` with:
- A table of 1.3B base / E1 / E2 × (FLORES es→gn, FLORES gn→es, jojajovai-anotado gn→es), with a ✅/❌ column per threshold: es→gn ≥ 36.05; gn→es ≥ 39.85; Jopara ≥ 49.31.
- What back-translation contributes (E2 − E1) per direction.
- The actual duration of each training run and the final parameters used.
- Recommendation: whether to move to full fine-tuning in the cloud, and why.

- [ ] **Step 5: Commit**

```bash
git add results/
git commit -m "docs: resultados E2 y conclusiones del subproyecto 1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
