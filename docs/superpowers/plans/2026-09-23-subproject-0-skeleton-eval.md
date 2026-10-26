# Subproject 0 — Skeleton + evaluation harness: Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python package, `guanani`, with a CLI that runs gn↔es NMT, gn STT and gn TTS with base models, plus a harness that produces reproducible baseline figures in `results/SUMMARY.md`.

**Architecture:**
- One thin module per component (`normalize`, `nmt`, `stt`, `tts`). Each exposes one function and takes the model id as a parameter.
- `data.py` reads local sets from `data/`.
- `eval.py` scores and writes JSON reports plus a row in `SUMMARY.md`.
- `cli.py` (argparse) wraps it all.
- No servers.

**Tech Stack:** Python ≥3.11, uv, torch, transformers (NLLB, MMS wav2vec2, VITS), sentencepiece, librosa, soundfile, sacrebleu, jiwer, unbabel-comet, huggingface_hub, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-subproject-0-skeleton-eval-design.md`

## Global Constraints

- Guarani ↔ Spanish only. No English/French.
- NLLB codes: `grn_Latn`, `spa_Latn`. FLORES+ uses `gug_Latn` for Guarani.
- Base models: `facebook/nllb-200-distilled-600M`, `facebook/nllb-200-distilled-1.3B`, `facebook/mms-1b-all` (`grn` adapter), `facebook/mms-tts-grn`.
- Internal audio at 16 kHz (`SR = 16000`).
- All Guarani text (input, output and references) goes through `normalize`.
- COMET into Guarani is reported as `comet_orientativo`; into Spanish as `comet`.
- Reports: `results/<date>_<component>_<model>_<dataset>[_<direction>].json` + a row in `results/SUMMARY.md`. `results/audio/` and `data/` are not versioned.
- Tests with real models are marked `@pytest.mark.slow` and excluded by default.
- Out of scope: fine-tuning, API/server, web, CTranslate2, human evaluation.
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Guarani text with variant spellings (`ꞌ`, `’`, `´`, `î`, `ÿ`, NFD) → hypotheses and references are identical after normalisation (Task 1).
2. Punctuation cleanup for WER/CER must not delete the *puso* or the combining tilde of `g̃` (Task 2).
3. Empty references in Common Voice, or ones that become empty after cleanup, and empty ASR hypotheses → must not break jiwer. They are excluded and counted (Task 2).
4. Empty or corrupt audio clip → skipped and counted in `skipped`, and evaluation continues (Task 6).
5. A model id with `/` (`facebook/...`) → produces a valid report file name inside `results/`, not a subdirectory (Task 6).

---

### Task 1: Package scaffolding + `normalize`

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/guanani/__init__.py`, `src/guanani/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Produces: `guanani.normalize.normalize(text: str) -> str`; constant `PUSO = "'"`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "guanani"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.4",
  "transformers>=4.45",
  "sentencepiece",
  "librosa>=0.10",
  "soundfile",
  "sacrebleu>=2.4",
  "jiwer>=3.0",
  "unbabel-comet>=2.2",
  "huggingface_hub>=0.34",
]

[project.scripts]
guanani = "guanani.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
markers = ["slow: carga modelos reales (descarga GB, lento)"]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Create `.gitignore` and `src/guanani/__init__.py` (empty)**

```gitignore
.venv/
__pycache__/
.pytest_cache/
data/
results/audio/
*.wav
```

- [ ] **Step 3: Install dependencies**

Run: `uv sync`
Expected: resolves and installs without error (downloads torch with CUDA, ~2-3 GB). If `unbabel-comet` clashes with the torch/transformers version, move `unbabel-comet` to an optional group `[project.optional-dependencies] comet = [...]` and note it in the commit.

- [ ] **Step 4: Write the failing tests**

`tests/test_normalize.py`:

```python
from guanani.normalize import normalize


def test_apostrophe_variants_become_puso():
    for v in "’‘´`ʼꞌ′‚":
        assert normalize(f"ka{v}i") == "ka'i"


def test_nfd_becomes_nfc():
    assert normalize("ãra") == "ãra"


def test_legacy_nasal_marks_from_jojajovai():
    assert normalize("omokyre'ÿ omopotî") == "omokyre'ỹ omopotĩ"
    assert normalize("Âva ära") == "Ãva ãra"


def test_g_tilde_variants_and_preserved():
    assert normalize("ĝuahẽ") == "g̃uahẽ"
    assert normalize("ḡuahẽ") == "g̃uahẽ"
    assert normalize("g̃uahẽ") == "g̃uahẽ"


def test_u_diaeresis_kept_for_spanish_loans():
    assert normalize("itaugüeño") == "itaugüeño"


def test_whitespace_and_zero_width():
    assert normalize("  che  ha​ʼe  \n") == "che ha'e"
```

- [ ] **Step 5: Run and verify it fails**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guanani.normalize'`

- [ ] **Step 6: Implement `src/guanani/normalize.py`**

```python
"""Normalización de texto guaraní: NFC, puso canónico y grafías nasales heredadas."""

import re
import unicodedata

PUSO = "'"

# ponytail: mapa fijo de grafías vistas en Jojajovai; ampliar si otro corpus trae variantes nuevas.
# 'ü' no se toca: aparece en préstamos del español (itaugüeño).
_TABLE = str.maketrans(
    {c: PUSO for c in "’‘´`ʼꞌ′‚"}
    | dict(zip("âêîôûŷÿäëïö", "ãẽĩõũỹỹãẽĩõ"))
    | dict(zip("ÂÊÎÔÛŶŸÄËÏÖ", "ÃẼĨÕŨỸỸÃẼĨÕ"))
    | {c: "g̃" for c in "ĝḡğġ"}
    | {c: "G̃" for c in "ĜḠĞĠ"}
    | {"​": None, "﻿": None}
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_TABLE)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
```

- [ ] **Step 7: Run and verify it passes**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/guanani/__init__.py src/guanani/normalize.py tests/test_normalize.py
git commit -m "feat: andamiaje del paquete y normalize para guaraní

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Metrics (`score_mt`, `score_comet`, `asr_clean`, `score_asr`, `by_group`)

**Files:**
- Create: `src/guanani/eval.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `guanani.normalize.normalize`
- Produces:
  - `score_mt(hyps: list[str], refs: list[str]) -> dict` with keys `"chrf++"`, `"bleu"` (floats, 2 decimals)
  - `score_comet(srcs, hyps, refs, model="Unbabel/wmt22-comet-da") -> float`
  - `asr_clean(text: str) -> str`
  - `score_asr(hyps, refs) -> dict` with `"wer"`, `"cer"` (4 decimals), `"empty_refs"` (int)
  - `by_group(hyps, refs, groups: list[str]) -> dict[str, dict]` → `{group: score_mt(...) | {"n": int}}`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
from guanani.eval import asr_clean, by_group, score_asr, score_mt


def test_score_mt_perfect_and_bad():
    assert score_mt(["che ha'e"], ["che ha'e"])["chrf++"] == 100.0
    assert score_mt(["xyz"], ["che ha'e"])["chrf++"] < 20


def test_score_mt_rewards_shared_morphemes():
    partial = score_mt(["ojapo hikuái"], ["ojapóta hikuái"])["chrf++"]
    assert 30 < partial < 100


def test_asr_clean_keeps_puso_and_combining_tilde():
    assert asr_clean("¡Ka’i, G̃uahẽ!") == "ka'i g̃uahẽ"


def test_score_asr_counts_deletion():
    s = score_asr(["che ha'e", "mba'e"], ["che ha'e", "mba'e porã"])
    assert s["wer"] == 0.25
    assert s["empty_refs"] == 0


def test_score_asr_skips_empty_refs_and_allows_empty_hyp():
    s = score_asr(["", "abc"], ["che", "..."])
    assert s["wer"] == 1.0
    assert s["empty_refs"] == 1


def test_by_group_splits_scores():
    out = by_group(["a b", "x"], ["a b", "c d"], ["guarani", "jopara"])
    assert out["guarani"]["chrf++"] == 100.0
    assert out["guarani"]["n"] == 1
    assert out["jopara"]["chrf++"] < 100.0
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guanani.eval'`

- [ ] **Step 3: Implement `src/guanani/eval.py` (metrics only)**

```python
"""Métricas e informes de evaluación."""

import re

import jiwer
import sacrebleu

from guanani.normalize import normalize


def score_mt(hyps: list[str], refs: list[str]) -> dict:
    return {
        "chrf++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
    }


def score_comet(srcs: list[str], hyps: list[str], refs: list[str], model: str = "Unbabel/wmt22-comet-da") -> float:
    import torch
    from comet import download_model, load_from_checkpoint

    m = load_from_checkpoint(download_model(model))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = m.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0, progress_bar=False)
    return round(out.system_score, 4)


def by_group(hyps: list[str], refs: list[str], groups: list[str]) -> dict:
    out = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        out[g] = score_mt([hyps[i] for i in idx], [refs[i] for i in idx]) | {"n": len(idx)}
    return out


def asr_clean(text: str) -> str:
    # \w no incluye marcas combinantes: se conserva U+0303 explícitamente (g̃).
    text = re.sub(r"[^\w\s'̃]", "", normalize(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def score_asr(hyps: list[str], refs: list[str]) -> dict:
    pairs = [(asr_clean(h), asr_clean(r)) for h, r in zip(hyps, refs)]
    kept = [(h, r) for h, r in pairs if r]
    hs, rs = [h for h, _ in kept], [r for _, r in kept]
    return {
        "wer": round(jiwer.wer(rs, hs), 4),
        "cer": round(jiwer.cer(rs, hs), 4),
        "empty_refs": len(pairs) - len(kept),
    }
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 6 passed. `test_score_asr_skips_empty_refs_and_allows_empty_hyp` may fail because this jiwer version rejects empty hypotheses. If so, replace empty hypotheses in `score_asr` with `"∅"` and re-run. `"∅"` is a token that never matches, so it counts as substitutions, which is equivalent to recognising nothing.

- [ ] **Step 5: Commit**

```bash
git add src/guanani/eval.py tests/test_metrics.py
git commit -m "feat: métricas chrF++/BLEU/COMET/WER/CER

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Data loading + download script

**Files:**
- Create: `src/guanani/data.py`, `scripts/download.sh`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `DataMissing(FileNotFoundError)`
  - `data_dir() -> Path` (env `GUANANI_DATA`, default `data`)
  - `load_flores(split="devtest") -> list[dict]` → `{"gn", "es"}`
  - `load_jojajovai(split="test") -> list[dict]` → `{"gn", "es", "source"}`
  - `load_jojajovai_annotated() -> list[dict]` → `{"gn", "es", "source", "dialect"}`
  - `load_common_voice(split="test") -> list[dict]` → `{"audio": str, "gn"}`
  - `DATASETS: dict[str, Callable[[], list[dict]]]` with keys `"flores"`, `"jojajovai"`, `"jojajovai-anotado"`, `"cv"`

On-disk format:
- `data/jojajovai/jojajovai_all.csv` — columns `split,source,gn,es,tokens_gn,tokens_es`
- `data/jojajovai/jojajovai_sample_annotations.csv` — columns `source,gn,es,...,dialect_unified,...`
- `data/flores/<split>.gug_Latn.jsonl` and `data/flores/<split>.spa_Latn.jsonl` — one JSON line per sentence with `id` and `text`
- `data/common_voice/<split>.tsv` (columns `path`, `sentence`, …) + `data/common_voice/clips/`

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:

```python
import json

import pytest

from guanani import data


@pytest.fixture
def ddir(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    return tmp_path


def test_missing_dataset_explains_how(ddir):
    with pytest.raises(data.DataMissing, match="download.sh"):
        data.load_jojajovai()


def test_jojajovai_filters_split(ddir):
    d = ddir / "jojajovai"
    d.mkdir()
    (d / "jojajovai_all.csv").write_text(
        "split,source,gn,es,tokens_gn,tokens_es\n"
        'train,abc,"ka\'i","mono",x,y\n'
        'test,abc,"mba\'e, porã","algo, bueno",x,y\n',
        encoding="utf-8",
    )
    assert data.load_jojajovai("test") == [{"gn": "mba'e, porã", "es": "algo, bueno", "source": "abc"}]


def test_jojajovai_annotated_has_dialect(ddir):
    d = ddir / "jojajovai"
    d.mkdir()
    (d / "jojajovai_sample_annotations.csv").write_text(
        "source,gn,es,dialect_unified\nblogs,che,yo,jopara\n", encoding="utf-8"
    )
    assert data.load_jojajovai_annotated() == [{"gn": "che", "es": "yo", "source": "blogs", "dialect": "jopara"}]


def test_flores_aligns_by_id(ddir):
    d = ddir / "flores"
    d.mkdir()
    (d / "devtest.gug_Latn.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "text": t}) for i, t in [(1, "peteĩ"), (2, "mokõi")]), encoding="utf-8"
    )
    (d / "devtest.spa_Latn.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "text": t}) for i, t in [(2, "dos"), (1, "uno")]), encoding="utf-8"
    )
    assert data.load_flores() == [{"gn": "peteĩ", "es": "uno"}, {"gn": "mokõi", "es": "dos"}]


def test_common_voice_tsv_with_quotes(ddir):
    d = ddir / "common_voice"
    d.mkdir()
    (d / "test.tsv").write_text('client_id\tpath\tsentence\nx\ta.mp3\tHe\'i "che"\n', encoding="utf-8")
    rows = data.load_common_voice()
    assert rows == [{"audio": str(d / "clips" / "a.mp3"), "gn": 'He\'i "che"'}]
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'data'`

- [ ] **Step 3: Implement `src/guanani/data.py`**

```python
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
        "Descarga Common Voice Guaraní (Scripted Speech, gn) desde "
        "https://datacollective.mozillafoundation.org y extrae test.tsv y clips/ en data/common_voice/ "
        "(ver scripts/download.sh)"
    ),
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
        return [{"audio": str(d / "clips" / r["path"]), "gn": r["sentence"]} for r in rows]


DATASETS = {
    "flores": load_flores,
    "jojajovai": load_jojajovai,
    "jojajovai-anotado": load_jojajovai_annotated,
    "cv": load_common_voice,
}
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Create `scripts/download.sh`** (and `chmod +x scripts/download.sh`)

```bash
#!/usr/bin/env bash
# Descarga los conjuntos de evaluación públicos a data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/jojajovai data/flores data/common_voice

base=https://raw.githubusercontent.com/pln-fing-udelar/jojajovai/main/data
for f in jojajovai_all.csv jojajovai_sample_annotations.csv; do
  curl -fsSL -o "data/jojajovai/$f" "$base/$f"
done
echo "Jojajovai OK"

uv run python - <<'EOF' || echo "FLORES+ falló: acepta las condiciones en https://huggingface.co/datasets/openlanguagedata/flores_plus y ejecuta 'uv run hf auth login'"
import shutil
from huggingface_hub import hf_hub_download
for split in ("dev", "devtest"):
    for code in ("gug_Latn", "spa_Latn"):
        p = hf_hub_download("openlanguagedata/flores_plus", f"{split}/{code}.jsonl", repo_type="dataset")
        shutil.copy(p, f"data/flores/{split}.{code}.jsonl")
print("FLORES+ OK")
EOF

cat <<'EOF'
Common Voice (manual, requiere cuenta):
  1. Descarga "Common Voice Scripted Speech — Guarani" desde https://datacollective.mozillafoundation.org
  2. Extrae y copia test.tsv (y dev.tsv si quieres) y la carpeta clips/ en data/common_voice/
EOF
```

- [ ] **Step 6: Run the real download**

Run: `./scripts/download.sh && wc -l data/jojajovai/*.csv data/flores/*`
Expected: `jojajovai_all.csv` ~30873 lines, annotations 401, each `devtest.*.jsonl` 1012 lines. If FLORES fails because access is restricted, ask the user to accept the conditions and run `! uv run hf auth login`, and carry on with Jojajovai meanwhile.

- [ ] **Step 7: Verify the loaders on real data**

Run: `uv run python -c "from guanani.data import *; print(len(load_jojajovai()), len(load_jojajovai_annotated()), len(load_flores()))"`
Expected: `5334 400 1012`

- [ ] **Step 8: Commit**

```bash
git add src/guanani/data.py scripts/download.sh tests/test_data.py
git commit -m "feat: loaders de FLORES+, Jojajovai y Common Voice + script de descarga

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Device selection + NMT (NLLB)

**Files:**
- Create: `src/guanani/device.py`, `src/guanani/nmt.py`
- Test: `tests/test_device.py`, `tests/test_models_slow.py`

**Interfaces:**
- Consumes: `normalize`
- Produces:
  - `guanani.device.pick_device(requested: str | None = None) -> str` (`"cuda"` or `"cpu"`; warns if cuda is requested but unavailable)
  - `guanani.nmt.DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"`
  - `guanani.nmt.translate(texts: list[str], src: str, tgt: str, model: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 16) -> list[str]` with `src`/`tgt` in `{"gn", "es"}`

- [ ] **Step 1: Write the failing device test**

`tests/test_device.py`:

```python
import pytest
import torch

from guanani.device import pick_device


def test_cuda_requested_but_missing_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CUDA"):
        assert pick_device("cuda") == "cpu"


def test_default_and_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device() == "cpu"
    assert pick_device("cpu") == "cpu"
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guanani.device'`

- [ ] **Step 3: Implement `src/guanani/device.py`**

```python
import warnings

import torch


def pick_device(requested: str | None = None) -> str:
    cuda = torch.cuda.is_available()
    if requested == "cuda" and not cuda:
        warnings.warn("CUDA no disponible; usando CPU")
        return "cpu"
    return requested or ("cuda" if cuda else "cpu")
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_device.py -v`
Expected: 2 passed

- [ ] **Step 5: Write the slow NMT test**

`tests/test_models_slow.py`:

```python
import pytest

pytestmark = pytest.mark.slow


def test_translate_both_directions():
    from guanani.nmt import translate

    es = translate(["Mba'éichapa reiko?"], "gn", "es")
    gn = translate(["Buenos días, ¿cómo estás?"], "es", "gn")
    assert len(es) == 1 and es[0].strip()
    assert len(gn) == 1 and gn[0].strip()
```

- [ ] **Step 6: Run and verify it fails**

Run: `uv run pytest -m slow tests/test_models_slow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guanani.nmt'`

- [ ] **Step 7: Implement `src/guanani/nmt.py`**

```python
"""Traducción gn↔es con NLLB."""

from functools import cache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from guanani.device import pick_device
from guanani.normalize import normalize

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
CODES = {"gn": "grn_Latn", "es": "spa_Latn"}


@cache
def _load(model: str, device: str):
    tok = AutoTokenizer.from_pretrained(model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    m = AutoModelForSeq2SeqLM.from_pretrained(model, torch_dtype=dtype).to(device).eval()
    return tok, m


def translate(
    texts: list[str],
    src: str,
    tgt: str,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 16,
) -> list[str]:
    device = pick_device(device)
    tok, m = _load(model, device)
    tok.src_lang = CODES[src]
    if src == "gn":
        texts = [normalize(t) for t in texts]
    bos = tok.convert_tokens_to_ids(CODES[tgt])
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.inference_mode():
            gen = m.generate(**enc, forced_bos_token_id=bos, max_new_tokens=256, num_beams=4)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return [normalize(o) for o in out] if tgt == "gn" else out
```

- [ ] **Step 8: Run and verify it passes**

Run: `uv run pytest -m slow tests/test_models_slow.py -v`
Expected: 1 passed (the first run downloads ~2.5 GB)

- [ ] **Step 9: Commit**

```bash
git add src/guanani/device.py src/guanani/nmt.py tests/test_device.py tests/test_models_slow.py
git commit -m "feat: selección de dispositivo y traducción gn↔es con NLLB

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: STT (MMS) + TTS (MMS-TTS)

**Files:**
- Create: `src/guanani/stt.py`, `src/guanani/tts.py`
- Test: `tests/test_audio.py`, modify `tests/test_models_slow.py`

**Interfaces:**
- Consumes: `pick_device`, `normalize`
- Produces:
  - `guanani.stt.SR = 16000`
  - `guanani.stt.DEFAULT_MODEL = "facebook/mms-1b-all"`
  - `guanani.stt.load_audio(path: str) -> np.ndarray` (float32 mono 16 kHz; `ValueError` if empty; other librosa exceptions if corrupt)
  - `guanani.stt.transcribe(audios: list[np.ndarray], model: str = DEFAULT_MODEL, device: str | None = None) -> list[str]`
  - `guanani.tts.DEFAULT_MODEL = "facebook/mms-tts-grn"`
  - `guanani.tts.speak(text: str, model: str = DEFAULT_MODEL, device: str | None = None) -> tuple[np.ndarray, int]` (`ValueError` if the text is empty)

- [ ] **Step 1: Write the failing fast tests**

`tests/test_audio.py`:

```python
import numpy as np
import pytest
import soundfile as sf

from guanani.stt import SR, load_audio


def test_load_audio_resamples_to_16k(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(22050, dtype=np.float32), 22050)
    audio = load_audio(str(p))
    assert abs(len(audio) - SR) <= 1


def test_load_audio_empty_raises(tmp_path):
    p = tmp_path / "empty.wav"
    sf.write(p, np.zeros(0, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="vacío"):
        load_audio(str(p))


def test_speak_rejects_empty_text():
    from guanani.tts import speak

    with pytest.raises(ValueError):
        speak("   ")
```

- [ ] **Step 2: Add slow tests to `tests/test_models_slow.py`**

```python
def test_tts_then_stt_roundtrip_nonempty():
    from guanani.stt import SR, transcribe
    from guanani.tts import speak

    audio, sr = speak("Mba'éichapa, che réra Juan.")
    assert sr == SR and audio.ndim == 1 and len(audio) > sr // 2
    text = transcribe([audio])[0]
    assert text.strip()
```

- [ ] **Step 3: Run and verify they fail**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guanani.stt'`

- [ ] **Step 4: Implement `src/guanani/stt.py`**

```python
"""Reconocimiento de voz en guaraní con MMS (adaptador grn)."""

from functools import cache

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from guanani.device import pick_device

SR = 16000
DEFAULT_MODEL = "facebook/mms-1b-all"


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if audio.size == 0:
        raise ValueError(f"audio vacío: {path}")
    return audio


@cache
def _load(model: str, device: str):
    proc = AutoProcessor.from_pretrained(model, target_lang="grn")
    m = Wav2Vec2ForCTC.from_pretrained(model, target_lang="grn", ignore_mismatched_sizes=True).to(device).eval()
    return proc, m


def transcribe(audios: list[np.ndarray], model: str = DEFAULT_MODEL, device: str | None = None) -> list[str]:
    device = pick_device(device)
    proc, m = _load(model, device)
    out = []
    # ponytail: de a uno, sin padding; agrupar en lotes si la evaluación de CV resulta lenta.
    for a in audios:
        inputs = proc(a, sampling_rate=SR, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = m(**inputs).logits.argmax(-1)[0]
        out.append(proc.decode(ids))
    return out
```

- [ ] **Step 5: Implement `src/guanani/tts.py`**

```python
"""Síntesis de voz en guaraní con MMS-TTS (VITS)."""

from functools import cache

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from guanani.device import pick_device
from guanani.normalize import normalize

DEFAULT_MODEL = "facebook/mms-tts-grn"


@cache
def _load(model: str, device: str):
    return AutoTokenizer.from_pretrained(model), VitsModel.from_pretrained(model).to(device).eval()


def speak(text: str, model: str = DEFAULT_MODEL, device: str | None = None) -> tuple[np.ndarray, int]:
    text = normalize(text)
    if not text:
        raise ValueError("texto vacío")
    device = pick_device(device)
    tok, m = _load(model, device)
    inputs = tok(text, return_tensors="pt").to(device)
    torch.manual_seed(0)  # VITS es estocástico; semilla fija para evaluaciones reproducibles
    with torch.inference_mode():
        wav = m(**inputs).waveform[0].float().cpu().numpy()
    return wav, m.config.sampling_rate
```

- [ ] **Step 6: Run and verify they pass**

Run: `uv run pytest tests/test_audio.py -v && uv run pytest -m slow tests/test_models_slow.py -v`
Expected: 3 passed; 2 passed (downloads ~4 GB of MMS-1B the first time)

- [ ] **Step 7: Commit**

```bash
git add src/guanani/stt.py src/guanani/tts.py tests/test_audio.py tests/test_models_slow.py
git commit -m "feat: STT con MMS-1B-all y TTS con MMS-TTS-grn

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Evaluation runner + reports

**Files:**
- Modify: `src/guanani/eval.py` (append at the end)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `data.DATASETS`, `nmt.translate`, `stt.load_audio`, `stt.transcribe`, `stt.SR`, `tts.speak`, `pick_device`, the Task 2 metrics
- Produces:
  - `COMPATIBLE: dict[str, set[str]]`
  - `load_clips(rows: list[dict]) -> tuple[list[np.ndarray], list[str], int]` (audios, references, skipped)
  - `write_report(report: dict, results_dir: Path) -> Path`
  - `run(component: str, dataset: str, model: str | None = None, direction: str = "gn-es", limit: int | None = None, device: str | None = None, comet: bool = True, results_dir: str = "results") -> Path`

Required `report` keys for `write_report`: `date`, `component`, `model`, `dataset`, `n`, `device`, `metrics` (dict). Optional: `direction`, `latency_s_per_item` or `rtf`.

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:

```python
import numpy as np
import pytest
import soundfile as sf

from guanani.eval import load_clips, run, write_report


def _report(**kw):
    base = {
        "date": "2026-09-23T1200",
        "component": "nmt",
        "model": "facebook/nllb-200-distilled-600M",
        "dataset": "flores",
        "direction": "gn-es",
        "n": 3,
        "device": "cpu",
        "metrics": {"chrf++": 40.1, "bleu": 9.5, "por_dialecto": {"jopara": {"chrf++": 30.0}}},
        "latency_s_per_item": 0.2,
    }
    return base | kw


def test_report_filename_has_no_slash_and_lives_in_results(tmp_path):
    path = write_report(_report(), tmp_path)
    assert path.parent == tmp_path
    assert "/" not in path.name and path.name.endswith(".json")
    assert "facebook-nllb-200-distilled-600M" in path.name


def test_summary_header_once_rows_appended(tmp_path):
    write_report(_report(), tmp_path)
    write_report(_report(date="2026-09-23T1300", rtf=0.05, latency_s_per_item=None), tmp_path)
    text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert text.count("| fecha |") == 1
    assert text.count("| nmt |") == 2
    assert "chrf++=40.1 bleu=9.5" in text
    assert "por_dialecto" not in text
    assert "RTF 0.05" in text


def test_load_clips_skips_broken_audio(tmp_path):
    good = tmp_path / "good.wav"
    sf.write(good, np.zeros(16000, dtype=np.float32), 16000)
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(0, dtype=np.float32), 16000)
    corrupt = tmp_path / "corrupt.mp3"
    corrupt.write_bytes(b"no es audio")
    rows = [{"audio": str(good), "gn": "a"}, {"audio": str(empty), "gn": "b"}, {"audio": str(corrupt), "gn": "c"}]
    audios, refs, skipped = load_clips(rows)
    assert len(audios) == 1 and refs == ["a"] and skipped == 2


def test_run_rejects_incompatible_dataset(tmp_path):
    with pytest.raises(ValueError, match="stt"):
        run("stt", "flores", results_dir=str(tmp_path))
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_clips'`

- [ ] **Step 3: Add to `src/guanani/eval.py`**

Add these imports at the very top (with the existing ones):

```python
import json
import time
from datetime import datetime
from pathlib import Path
```

And at the end of the file:

```python
SAMPLES = 20
AUDIO_SAMPLES = 10
COMPATIBLE = {
    "nmt": {"flores", "jojajovai", "jojajovai-anotado"},
    "stt": {"cv"},
    "tts": {"flores", "jojajovai", "jojajovai-anotado", "cv"},
}
HEADER = (
    "| fecha | componente | modelo | dataset | dirección | n | métricas | latencia | dispositivo |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def write_report(report: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    parts = [report["date"], report["component"], report["model"], report["dataset"], report.get("direction") or ""]
    path = results_dir / ("_".join(_slug(p) for p in parts if p) + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = results_dir / "SUMMARY.md"
    if not summary.exists():
        summary.write_text("# Resultados\n\n" + HEADER, encoding="utf-8")
    metrics = " ".join(f"{k}={v}" for k, v in report["metrics"].items() if not isinstance(v, dict))
    lat = report.get("latency_s_per_item")
    latency = f"{lat} s/frase" if lat is not None else f"RTF {report.get('rtf')}"
    row = (
        f"| {report['date']} | {report['component']} | {report['model']} | {report['dataset']} "
        f"| {report.get('direction') or ''} | {report['n']} | {metrics} | {latency} | {report['device']} |\n"
    )
    with summary.open("a", encoding="utf-8") as f:
        f.write(row)
    return path


def load_clips(rows: list[dict]):
    from guanani.stt import load_audio

    audios, refs, skipped = [], [], 0
    for r in rows:
        try:
            audios.append(load_audio(r["audio"]))
            refs.append(r["gn"])
        except Exception:
            skipped += 1
    return audios, refs, skipped


def _eval_nmt(rows, model, direction, device, comet):
    from guanani.nmt import translate

    src, tgt = direction.split("-")
    srcs = [r[src] for r in rows]
    refs = [normalize(r[tgt]) if tgt == "gn" else r[tgt] for r in rows]
    t = time.perf_counter()
    hyps = translate(srcs, src, tgt, model=model, device=device)
    secs = time.perf_counter() - t
    metrics = score_mt(hyps, refs)
    if comet:
        metrics["comet" if tgt == "es" else "comet_orientativo"] = score_comet(srcs, hyps, refs)
    if "dialect" in rows[0]:
        metrics["por_dialecto"] = by_group(hyps, refs, [r["dialect"] for r in rows])
    return {
        "direction": direction,
        "n": len(rows),
        "metrics": metrics,
        "latency_s_per_item": round(secs / len(rows), 4),
        "samples": [{"src": s, "ref": r, "hyp": h} for s, r, h in list(zip(srcs, refs, hyps))[:SAMPLES]],
    }


def _eval_stt(rows, model, device):
    from guanani.stt import SR, transcribe

    audios, refs, skipped = load_clips(rows)
    if not audios:
        raise RuntimeError("Ningún clip de audio válido")
    t = time.perf_counter()
    hyps = transcribe(audios, model=model, device=device)
    secs = time.perf_counter() - t
    duration = sum(len(a) for a in audios) / SR
    return {
        "n": len(audios),
        "skipped": skipped,
        "metrics": score_asr(hyps, refs),
        "rtf": round(secs / duration, 4),
        "samples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:SAMPLES]],
    }


def _eval_tts(rows, model, device, audio_dir: Path):
    import librosa
    import soundfile as sf

    from guanani.stt import DEFAULT_MODEL as ASR_MODEL
    from guanani.stt import SR, transcribe
    from guanani.tts import speak

    texts = [t for t in (normalize(r["gn"]) for r in rows) if t]
    wavs = []
    t = time.perf_counter()
    for text in texts:
        a, sr = speak(text, model=model, device=device)
        wavs.append(a if sr == SR else librosa.resample(a, orig_sr=sr, target_sr=SR))
    secs = time.perf_counter() - t
    duration = sum(len(w) for w in wavs) / SR
    hyps = transcribe(wavs, model=ASR_MODEL, device=device)
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(wavs[:AUDIO_SAMPLES]):
        sf.write(audio_dir / f"{i:02d}.wav", w, SR)
    return {
        "n": len(texts),
        "asr_model": ASR_MODEL,
        "audio_dir": str(audio_dir),
        "metrics": {f"roundtrip_{k}": v for k, v in score_asr(hyps, texts).items()},
        "rtf": round(secs / duration, 4),
        "samples": [{"text": x, "asr": h} for x, h in list(zip(texts, hyps))[:SAMPLES]],
    }


def run(
    component: str,
    dataset: str,
    model: str | None = None,
    direction: str = "gn-es",
    limit: int | None = None,
    device: str | None = None,
    comet: bool = True,
    results_dir: str = "results",
) -> Path:
    if dataset not in COMPATIBLE.get(component, set()):
        raise ValueError(f"{component} no admite el dataset {dataset}; opciones: {sorted(COMPATIBLE.get(component, []))}")
    from guanani import data
    from guanani.device import pick_device

    rows = data.DATASETS[dataset]()[:limit]
    device = pick_device(device)
    date = datetime.now().strftime("%Y-%m-%dT%H%M")
    if component == "nmt":
        from guanani.nmt import DEFAULT_MODEL

        model = model or DEFAULT_MODEL
        report = _eval_nmt(rows, model, direction, device, comet)
    elif component == "stt":
        from guanani.stt import DEFAULT_MODEL

        model = model or DEFAULT_MODEL
        report = _eval_stt(rows, model, device)
    else:
        from guanani.tts import DEFAULT_MODEL

        model = model or DEFAULT_MODEL
        report = _eval_tts(rows, model, device, Path(results_dir) / "audio" / f"{date}_{_slug(model)}")
    report |= {"date": date, "component": component, "model": model, "dataset": dataset, "device": device, "limit": limit}
    return write_report(report, Path(results_dir))
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: 4 passed

- [ ] **Step 5: Short real run**

Run: `uv run python -c "from guanani.eval import run; print(run('nmt','jojajovai-anotado',limit=20,comet=False,results_dir='/tmp/guanani-smoke'))" && cat /tmp/guanani-smoke/SUMMARY.md`
Expected: a JSON file is created and a row with `chrf++=… bleu=…` appears; the JSON contains `por_dialecto`.

- [ ] **Step 6: Run the whole fast suite**

Run: `uv run pytest -v`
Expected: all pass (the `slow` tests are deselected)

- [ ] **Step 7: Commit**

```bash
git add src/guanani/eval.py tests/test_report.py
git commit -m "feat: runner de evaluación con informes JSON y SUMMARY.md

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: CLI

**Files:**
- Create: `src/guanani/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `translate`, `transcribe`, `load_audio`, `speak`, `eval.run`, `data.DATASETS`, `data.DataMissing`
- Produces: `guanani.cli.main(argv: list[str] | None = None) -> int` (0 ok, 2 data/usage error); the `guanani` command.

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:

```python
from guanani.cli import main


def test_eval_missing_data_exits_2_with_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    assert main(["eval", "nmt", "flores", "--no-comet"]) == 2
    assert "Falta" in capsys.readouterr().err


def test_eval_incompatible_dataset_exits_2(capsys):
    assert main(["eval", "stt", "flores"]) == 2
    assert "stt" in capsys.readouterr().err
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'guanani.cli'`

- [ ] **Step 3: Implement `src/guanani/cli.py`**

```python
"""CLI: guanani translate|transcribe|speak|pipeline|eval."""

import argparse
import sys

from guanani.data import DATASETS, DataMissing


def _kw(a) -> dict:
    return {"model": a.model} if getattr(a, "model", None) else {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="guanani")
    p.add_argument("--device", choices=["cuda", "cpu"])
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="traduce texto gn↔es")
    t.add_argument("text")
    t.add_argument("--src", choices=["gn", "es"], default="gn")
    t.add_argument("--tgt", choices=["gn", "es"], default="es")
    t.add_argument("--model")

    s = sub.add_parser("transcribe", help="voz guaraní → texto")
    s.add_argument("audio")
    s.add_argument("--model")

    k = sub.add_parser("speak", help="texto guaraní → voz")
    k.add_argument("text")
    k.add_argument("-o", "--out", default="out.wav")
    k.add_argument("--model")

    pl = sub.add_parser("pipeline", help="voz gn → texto es, o texto es → voz gn")
    g = pl.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio", help="audio en guaraní → traducción al español")
    g.add_argument("--text", help="texto en español → voz en guaraní")
    pl.add_argument("-o", "--out", default="out.wav")

    e = sub.add_parser("eval", help="evalúa un componente y escribe el informe en results/")
    e.add_argument("component", choices=["nmt", "stt", "tts"])
    e.add_argument("dataset", choices=list(DATASETS))
    e.add_argument("--direction", choices=["gn-es", "es-gn"], default="gn-es")
    e.add_argument("--model")
    e.add_argument("--limit", type=int)
    e.add_argument("--no-comet", action="store_true")

    a = p.parse_args(argv)
    try:
        if a.cmd == "translate":
            from guanani.nmt import translate

            print(translate([a.text], a.src, a.tgt, device=a.device, **_kw(a))[0])
        elif a.cmd == "transcribe":
            from guanani.stt import load_audio, transcribe

            print(transcribe([load_audio(a.audio)], device=a.device, **_kw(a))[0])
        elif a.cmd == "speak":
            import soundfile as sf

            from guanani.tts import speak

            audio, sr = speak(a.text, device=a.device, **_kw(a))
            sf.write(a.out, audio, sr)
            print(a.out)
        elif a.cmd == "pipeline":
            from guanani.nmt import translate

            if a.audio:
                from guanani.stt import load_audio, transcribe

                gn = transcribe([load_audio(a.audio)], device=a.device)[0]
                print(f"gn: {gn}\nes: {translate([gn], 'gn', 'es', device=a.device)[0]}")
            else:
                import soundfile as sf

                from guanani.tts import speak

                gn = translate([a.text], "es", "gn", device=a.device)[0]
                audio, sr = speak(gn, device=a.device)
                sf.write(a.out, audio, sr)
                print(f"gn: {gn}\naudio: {a.out}")
        else:
            from guanani.eval import run

            path = run(a.component, a.dataset, direction=a.direction, limit=a.limit,
                       device=a.device, comet=not a.no_comet, **_kw(a))
            print(path)
    except (DataMissing, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Try the real commands**

Run:
```bash
uv run guanani translate "Mba'éichapa reiko?"
uv run guanani translate "Buenos días" --src es --tgt gn
uv run guanani speak "Mba'éichapa" -o /tmp/g.wav
uv run guanani transcribe /tmp/g.wav
uv run guanani pipeline --audio /tmp/g.wav
uv run guanani pipeline --text "Buenos días, ¿cómo estás?" -o /tmp/p.wav
```
Expected: each command prints text or a path without error; `/tmp/g.wav` and `/tmp/p.wav` sound like Guarani (`aplay /tmp/p.wav`).

- [ ] **Step 6: Commit**

```bash
git add src/guanani/cli.py tests/test_cli.py
git commit -m "feat: CLI guanani translate|transcribe|speak|pipeline|eval

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Baseline figures and conclusions

**Files:**
- Create: `results/SUMMARY.md` (generated by `eval`), `results/*.json`, `README.md`

**Interfaces:**
- Consumes: the `guanani eval` CLI.

- [ ] **Step 1: Check the Common Voice data**

Run: `ls data/common_voice/test.tsv data/common_voice/clips | head -3`
Expected: they exist. If not, ask the user to download them (instructions in `scripts/download.sh`) and carry on with NMT/TTS meanwhile.

- [ ] **Step 2: NMT on GPU (12 runs)**

Run:
```bash
for m in facebook/nllb-200-distilled-600M facebook/nllb-200-distilled-1.3B; do
  for d in gn-es es-gn; do
    for ds in flores jojajovai jojajovai-anotado; do
      uv run guanani eval nmt $ds --direction $d --model $m
    done
  done
done
```
Expected: 12 JSON files in `results/` and 12 rows in `SUMMARY.md`. If loading COMET alongside the 1.3B runs out of VRAM, repeat that run with `--no-comet` and note it.

- [ ] **Step 3: STT and TTS on GPU**

Run:
```bash
uv run guanani eval stt cv
uv run guanani eval tts flores --limit 200
```
Expected: rows with `wer/cer` and `roundtrip_wer/roundtrip_cer`; 10 WAVs in `results/audio/…`.

- [ ] **Step 4: CPU latency (VPS scenario)**

Run:
```bash
for m in facebook/nllb-200-distilled-600M facebook/nllb-200-distilled-1.3B; do
  uv run guanani --device cpu eval nmt flores --model $m --limit 50 --no-comet
done
uv run guanani --device cpu eval stt cv --limit 50
uv run guanani --device cpu eval tts flores --limit 20
```
Expected: rows with device `cpu`.

- [ ] **Step 5: Write conclusions in `results/SUMMARY.md`**

Insert a `## Conclusions` section between `# Results` and the table. It must not go at the end, because `eval` appends rows there. Cover:
- the best NMT model per direction (chrF++);
- the gap between Guarani and Jopara/jehe'a in the annotated sample;
- the STT CER;
- the TTS round-trip CER compared with the STT CER (the TTS is only bad if its round-trip CER clearly exceeds the STT's);
- CPU latencies against the interactive target (< 1 s per sentence);
- **the weakest component and the next subproject**.

- [ ] **Step 6: Create `README.md`**

````markdown
# Guanani

Traducción guaraní ↔ español, reconocimiento y síntesis de voz en guaraní.

## Uso

```bash
uv sync
./scripts/download.sh            # Jojajovai + FLORES+ (Common Voice es manual)
uv run guanani translate "Mba'éichapa reiko?"
uv run guanani speak "Mba'éichapa" -o hola.wav
uv run guanani transcribe hola.wav
uv run guanani pipeline --text "Buenos días" -o saludo.wav
uv run guanani eval nmt flores --direction gn-es
```

Results in `results/SUMMARY.md`. Tests: `uv run pytest` (fast) and `uv run pytest -m slow` (real models).

NLLB-200 and MMS base models: CC-BY-NC-4.0 licence (non-commercial use only).
````

- [ ] **Step 7: Commit**

```bash
git add results/ README.md
git commit -m "docs: cifras base del subproyecto 0 y conclusiones

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
