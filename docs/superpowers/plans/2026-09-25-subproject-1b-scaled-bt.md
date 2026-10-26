# Subproject 1b — Scaled back-translation, GlotLID and tagged BT: Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 500,000 back-translated sentences (gn Wikipedia + KuatIA), filtered by GlotLID and sampled with E2. Then train E3 (no tag) and E4 (with `[BT] `) on the user's server and measure them against the thresholds.

**Architecture:** The existing pipeline is extended:
- `corpus.py` replaces `looks_paraguayan_guarani`/`sample_sentences` with `passes_text_rules` + `clean_sentences` (with statistics and an injectable language filter) + `reservoir`.
- New `lid.py` (GlotLID).
- `sources.py` adds Wikipedia.
- `nmt.translate` gains sampling.
- `backtranslate.py` moves to named runs in `data/train/<name>/`.
- `lora.py` generalises the data files (`btN`) and adds upsampling and the tag.

The heavy runs go to `user@gpu-server`.

**Tech Stack:** Python 3.12, uv, torch, transformers 4.57, peft, fasttext-wheel (GlotLID), pandas/pyarrow, sacrebleu, pytest, rsync/ssh.

**Spec:** `docs/superpowers/specs/2026-09-25-subproject-1b-scaled-bt-design.md`

## Global Constraints

- **Thresholds** (FLORES+ devtest except the third): es→gn ≥ 36.05; gn→es ≥ 39.85; `jojajovai-anotado` gn→es ≥ 49.31. Also report BLEU.
- **GlotLID:** `cis-lmu/glotlid` `model_v3.bin`. Accepted labels: `gug_Latn`, `grn_Latn`. The threshold is calibrated to let through ≥ 95 % of the gn sides of `data/train/parallel.jsonl`.
- **Rules that stay:** Spanish stop words ≤ 25 %, words with digits ≤ 20 %, 5–60 words. The spelling rule (`ɨʉwq`, `nh`, `kuery`) is removed.
- **`bt2`:**
  - all valid Wikipedia sentences (`wikimedia/wikipedia`, `20231101.gn`) + a KuatIA sample up to 500,000;
  - seed 42; sampling with `top_k=10`, `num_beams=1` using `models/e2`; blocks of 1,000;
  - everything in `data/train/bt2/`; `data/train/bt.jsonl` is left untouched.
- **E3:** `--data parallel,bt2 --upsample-parallel 4 --epochs 1 --eval-steps 2000 --batch-size 16 --grad-accum 2`. **E4:** the same + `--bt-tag` (prefix `[BT] `). Both start from `facebook/nllb-200-distilled-1.3B`.
- **Server:** `user@gpu-server`, repo in `~/Guanani`, `uv` in `~/.local/bin`.
  - Launch long jobs with `setsid nohup … < /dev/null &`.
  - There is no HF login there, so gated data (KuatIA, FLORES, Common Voice) is copied over from local with `rsync`.
- The code path that generated E2's `bt.jsonl` (unnamed `backtranslate()`) goes away. The file is kept and `--data parallel,bt` still works.
- `pkill` with patterns that do not match the command itself (`"[n]ame e3"`).
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A sentence rejected by GlotLID must not block a different sentence. An accepted sentence must not be duplicated, even if it comes from both Wikipedia and KuatIA (Task 1, Task 4).
2. If there are fewer than 500,000 candidates, `bt2` is generated from what exists without error, and `stats.json` reflects it (Task 4).
3. Relaunching `backtranslate --name bt2` with different settings → an error, never a mix of blocks (Task 4).
4. The `[BT] ` tag never appears in real-pair examples or at inference (Task 1, Task 5).
5. `--data parallel,bt2` without `bt2.jsonl` → `DataMissing` before loading any model (Task 5).

---

### Task 1: Corpus functions: rules, cleaning with statistics, reservoir and tag

**Files:**
- Modify: `src/guanani/train/corpus.py`
- Modify: `tests/test_corpus.py`

**Interfaces:**
- Produces:
  - `BT_TAG = "[BT] "`
  - `passes_text_rules(sentence: str) -> bool`
  - `clean_sentences(docs: Iterable[tuple[str, str]], eval_keys: set[str], stats: Counter, seen: set[str], reject_lang: Callable[[str], str | None] | None = None, min_words: int = 5, max_words: int = 60) -> Iterator[tuple[str, str]]` — `stats` keys: `"<source>:longitud"`, `":reglas"`, `":contaminada"`, `":duplicada"`, `":lid:<label>"`, `":aceptada"`
  - `reservoir(items: Iterable, n: int, seed: int = 42) -> list`
  - `build_examples(parallel, bt=(), bt_tag: bool = False) -> list[dict]`
- Removes: `looks_paraguayan_guarani`, `sample_sentences`, `_OTHER_VARIETY` (their uses are replaced in Task 4).

- [ ] **Step 1: Rewrite the affected tests in `tests/test_corpus.py`**

Change the import block to:

```python
from collections import Counter

from guanani.train.corpus import (
    KUATIA_SUBSETS,
    build_examples,
    clean_pairs,
    clean_sentences,
    decont_key,
    passes_text_rules,
    reservoir,
    split_sentences,
)
```

Delete `test_sample_sentences_filters_dedups_decontaminates_and_is_deterministic`, `test_looks_paraguayan_guarani_rejects_other_varieties_spanish_and_numbers` and `test_sample_sentences_skips_non_paraguayan_guarani`. Append:

```python
def test_passes_text_rules_rejects_spanish_and_numbers_keeps_loans():
    assert passes_text_rules("Tetãvore Tarija ha'e peteĩ tetãvore ñemby gotyo Volíviape.")
    assert passes_text_rules("Ojeporu web ha whatsapp mbo'ehaópe opa ára.")
    assert not passes_text_rules("El gobierno de la ciudad anunció que las obras terminan en marzo.")
    assert not passes_text_rules("- 00:3000:30 10 jasyteĩ 2012 difs.")


def test_clean_sentences_counts_each_reason_per_source():
    docs = [(
        "wiki",
        "Mbykymi. Ko ñe'ẽ ipuku porã ha iporã. Ko ñe'ẽ ipuku porã ha iporã. "
        "El gobierno de la ciudad anunció que las obras terminan. "
        "Ambue ñe'ẽ ipuku porã avei ko'ápe. Ha'e ramo nhuã rarõa omboaxa pova'e kuery.",
    )]
    keys = {decont_key("Ambue ñe'ẽ ipuku porã avei ko'ápe.")}
    stats, seen = Counter(), set()
    got = list(clean_sentences(docs, keys, stats, seen, lambda s: "gun_Latn" if "nhuã" in s else None))
    assert got == [("wiki", "Ko ñe'ẽ ipuku porã ha iporã.")]
    assert stats == Counter({
        "wiki:longitud": 1, "wiki:aceptada": 1, "wiki:duplicada": 1,
        "wiki:reglas": 1, "wiki:contaminada": 1, "wiki:lid:gun_Latn": 1,
    })


def test_clean_sentences_dedups_across_calls_sharing_seen():
    stats, seen = Counter(), set()
    doc = "Ko ñe'ẽ ipuku porã ha iporã."
    assert len(list(clean_sentences([("wikipedia", doc)], set(), stats, seen))) == 1
    assert list(clean_sentences([("madlad400", doc)], set(), stats, seen)) == []
    assert stats["madlad400:duplicada"] == 1


def test_reservoir_deterministic_and_bounded():
    a = reservoir(range(100), 10, seed=3)
    assert len(a) == 10 and a == reservoir(range(100), 10, seed=3)
    assert reservoir(range(5), 10) == [0, 1, 2, 3, 4]
    assert reservoir(range(5), 0) == []


def test_build_examples_bt_tag_only_on_synthetic():
    ex = build_examples([_p("che", "yo")], bt=[_p("ore", "nosotros", "bt:x")], bt_tag=True)
    assert [e["src"] for e in ex] == ["che", "yo", "[BT] nosotros"]
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_corpus.py -q`
Expected: FAIL with `ImportError: cannot import name 'clean_sentences'`

- [ ] **Step 3: Modify `src/guanani/train/corpus.py`**

1. Change the imports to:

```python
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator

from guanani.normalize import normalize
```

2. After `KUATIA_SUBSETS`, add: `BT_TAG = "[BT] "`.

3. Replace the `# ponytail: heurística barata…` comment, `_OTHER_VARIETY` and the `looks_paraguayan_guarani` function with:

```python
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
```

4. Replace the whole `sample_sentences` function with:

```python
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
```

5. Replace `build_examples` with:

```python
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
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_corpus.py -q`
Expected: 12 passed. (`tests/test_backtranslate.py` will fail to import until Task 4; that is expected.)

- [ ] **Step 5: Commit**

```bash
git add src/guanani/train/corpus.py tests/test_corpus.py
git commit -m "feat: limpieza de frases con estadísticas, reservorio y etiqueta [BT]

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: GlotLID variety identifier

**Files:**
- Modify: `pyproject.toml`
- Create: `src/guanani/train/lid.py`
- Test: `tests/test_lid.py`

**Interfaces:**
- Produces:
  - `ACCEPTED = frozenset({"gug_Latn", "grn_Latn"})`
  - `load_glotlid()` → fastText model (a clear `RuntimeError` if `fasttext` is missing)
  - `class GlotLID(model, threshold: float = 0.0)` with:
    - `top(s) -> tuple[str, float]`;
    - `accepted_prob(s) -> float`, which is 0.0 if the top-1 label is not accepted;
    - `reject_label(s) -> str | None`: `None` means accepted; otherwise the rival label, or `"<label><threshold"` if the label is accepted but below the threshold.
  - `calibrate_threshold(probs: list[float], recall: float = 0.95) -> float`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, inside `dependencies`, add `"fasttext-wheel>=0.9.2",`.

Run: `UV_HTTP_TIMEOUT=300 uv sync -q && uv run python -c "import fasttext; print('ok')"`
Expected: `ok`

- [ ] **Step 2: Write the failing tests**

`tests/test_lid.py`:

```python
from guanani.train.lid import GlotLID, calibrate_threshold


class _Fake:
    def __init__(self, table):
        self.table = table

    def predict(self, s, k=1):
        label, p = self.table[s]
        return (f"__label__{label}",), [p]


def test_reject_label_and_accepted_prob():
    lid = GlotLID(_Fake({"a": ("gug_Latn", 0.99), "b": ("gun_Latn", 1.0), "c": ("grn_Latn", 0.4), "d": ("spa_Latn", 0.9)}), 0.5)
    assert [lid.reject_label(s) for s in "abcd"] == [None, "gun_Latn", "grn_Latn<umbral", "spa_Latn"]
    assert lid.accepted_prob("a") == 0.99 and lid.accepted_prob("b") == 0.0


def test_calibrate_threshold_keeps_95_percent():
    probs = [1.0] * 90 + [0.8] * 5 + [0.3] * 4 + [0.0]
    t = calibrate_threshold(probs)
    assert t == 0.8
    assert sum(p >= t for p in probs) / len(probs) >= 0.95
```

- [ ] **Step 3: Run and verify it fails**

Run: `uv run pytest tests/test_lid.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'guanani.train.lid'`

- [ ] **Step 4: Implement `src/guanani/train/lid.py`**

```python
"""Identificación de variedad con GlotLID: acepta guaraní paraguayo, rechaza otras variedades y lenguas."""

import math

ACCEPTED = frozenset({"gug_Latn", "grn_Latn"})
GLOTLID = ("cis-lmu/glotlid", "model_v3.bin")


def load_glotlid():
    try:
        import fasttext
    except ImportError as e:
        raise RuntimeError("Falta fasttext: ejecuta `uv sync` (dependencia fasttext-wheel)") from e
    from huggingface_hub import hf_hub_download

    return fasttext.load_model(hf_hub_download(*GLOTLID))


class GlotLID:
    def __init__(self, model, threshold: float = 0.0):
        self.model, self.threshold = model, threshold

    def top(self, sentence: str) -> tuple[str, float]:
        labels, probs = self.model.predict(sentence.replace("\n", " "), k=1)
        return labels[0].removeprefix("__label__"), float(probs[0])

    def accepted_prob(self, sentence: str) -> float:
        label, p = self.top(sentence)
        return p if label in ACCEPTED else 0.0

    def reject_label(self, sentence: str) -> str | None:
        label, p = self.top(sentence)
        if label not in ACCEPTED:
            return label
        return None if p >= self.threshold else f"{label}<umbral"


def calibrate_threshold(probs: list[float], recall: float = 0.95) -> float:
    # Mayor umbral que deja pasar al menos `recall` de las frases de referencia.
    ranked = sorted(probs, reverse=True)
    return ranked[max(0, math.ceil(recall * len(ranked)) - 1)]
```

- [ ] **Step 5: Run and verify it passes**

Run: `uv run pytest tests/test_lid.py -q`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/guanani/train/lid.py tests/test_lid.py
git commit -m "feat: filtro de variedad con GlotLID y calibración del umbral

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Guarani Wikipedia as a source

**Files:**
- Modify: `src/guanani/data.py` (key in `HOW`), `src/guanani/train/sources.py`, `scripts/download.sh`
- Test: `tests/test_sources.py` (append)

**Interfaces:**
- Produces: `sources.iter_wikipedia_docs() -> Iterator[tuple[str, str]]` (source `"wikipedia"`), reading `data/wikipedia/gn.parquet` (column `text`).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sources.py`)

```python
def test_wikipedia_docs_from_parquet(ddir):
    import pandas as pd

    (ddir / "wikipedia").mkdir()
    pd.DataFrame({"title": ["A", "B", "C"], "text": ["Ñe'ẽ peteĩ.", "", "Ñe'ẽ mokõi."]}).to_parquet(
        ddir / "wikipedia" / "gn.parquet"
    )
    assert list(sources.iter_wikipedia_docs()) == [("wikipedia", "Ñe'ẽ peteĩ."), ("wikipedia", "Ñe'ẽ mokõi.")]


def test_wikipedia_missing_explains_how(ddir):
    with pytest.raises(data.DataMissing, match="Wikipedia"):
        next(sources.iter_wikipedia_docs())
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_sources.py -q`
Expected: FAIL with `AttributeError: module 'guanani.train.sources' has no attribute 'iter_wikipedia_docs'`

- [ ] **Step 3: Implement**

In `src/guanani/data.py`, inside `HOW`, add:

```python
    "wikipedia": "Ejecuta scripts/download.sh (Wikipedia en guaraní, wikimedia/wikipedia 20231101.gn)",
```

In `src/guanani/train/sources.py`, after `iter_kuatia_docs`, add:

```python
def iter_wikipedia_docs() -> Iterator[tuple[str, str]]:
    import pandas as pd

    p = data._require(data.data_dir() / "wikipedia" / "gn.parquet", "wikipedia")
    for text in pd.read_parquet(p, columns=["text"])["text"]:
        if isinstance(text, str) and text.strip():
            yield "wikipedia", text
```

The missing-file test works because the generator runs `_require` on the first `next()`.

At the end of `scripts/download.sh`, add:

```bash
uv run python - <<'EOF' || echo "Wikipedia gn falló"
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
Path("data/wikipedia").mkdir(parents=True, exist_ok=True)
p = hf_hub_download("wikimedia/wikipedia", "20231101.gn/train-00000-of-00001.parquet", repo_type="dataset")
shutil.copy(p, "data/wikipedia/gn.parquet")
print("Wikipedia gn OK")
EOF
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest tests/test_sources.py -q`
Expected: 7 passed

- [ ] **Step 5: Download and check the real Wikipedia**

Run: `UV_HTTP_TIMEOUT=300 ./scripts/download.sh 2>&1 | grep -E "OK|falló"; uv run python -c "from guanani.train.sources import iter_wikipedia_docs as w; d=list(w()); print(len(d)); print(d[0][1][:200])"`
Expected: `Wikipedia gn OK` and about 5,000–6,000 documents; the first is plain Guarani text. If the columns do not include `text`, adjust `iter_wikipedia_docs` and record a `Ruling:`.

- [ ] **Step 6: Commit**

```bash
git add src/guanani/data.py src/guanani/train/sources.py scripts/download.sh tests/test_sources.py
git commit -m "feat: Wikipedia en guaraní como fuente de frases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Sampled translation and named back-translation

**Files:**
- Modify: `src/guanani/nmt.py`, `src/guanani/train/backtranslate.py` (rewrite)
- Modify: `tests/test_nmt.py` (append), `tests/test_backtranslate.py` (rewrite)

**Interfaces:**
- Consumes: `corpus.clean_sentences`, `corpus.reservoir`, `lid.*`, `sources.eval_keys`, `sources.iter_kuatia_docs`, `sources.iter_wikipedia_docs`, `sources.read_jsonl`, `sources.write_jsonl`, `sources.train_dir`, `data._require`
- Produces:
  - `nmt.translate(texts, src, tgt, model=DEFAULT_MODEL, device=None, batch_size=16, sample=False)`. With `sample=True`: `do_sample=True`, `top_k=10`, `num_beams=1`, and `torch.manual_seed(0)` per call.
  - `backtranslate.select_sentences(n: int, seed: int = 42, lid: bool = True) -> tuple[list[dict], dict]` — rows `{gn, source}`; stats `{n, lid_threshold, counts}`
  - `backtranslate.backtranslate(name: str, n: int = 500000, model: str = BT_MODEL, sample: bool = False, batch_size: int = 16, chunk: int = 1000, seed: int = 42, lid: bool = True) -> Path` — writes `data/train/<name>/{meta.json, stats.json, source.jsonl, parts/, <name>.jsonl}`

- [ ] **Step 1: Sampling test (append to `tests/test_nmt.py`)**

```python
class _RecordsKwargs:
    def __init__(self):
        self.kwargs = []

    def generate(self, texts, **kw):
        self.kwargs.append(kw)
        return texts


def test_sample_uses_topk_sampling_and_default_uses_beam(monkeypatch):
    m = _RecordsKwargs()
    monkeypatch.setattr(nmt, "_load", lambda model, device: (_Tok(), m))
    nmt.translate(["a"], "es", "es", device="cpu", sample=True)
    nmt.translate(["a"], "es", "es", device="cpu")
    assert (m.kwargs[0]["do_sample"], m.kwargs[0]["top_k"], m.kwargs[0]["num_beams"]) == (True, 10, 1)
    assert m.kwargs[1]["num_beams"] == 4 and "do_sample" not in m.kwargs[1]
```

- [ ] **Step 2: Rewrite `tests/test_backtranslate.py`**

```python
import pytest

from guanani import nmt
from guanani.train import backtranslate as bt
from guanani.train import sources


@pytest.fixture
def ddir(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "bt2" / "source.jsonl",
                        [{"gn": f"frase {i}", "source": "madlad400"} for i in range(25)])
    (tmp_path / "train" / "bt.jsonl").write_text("sentinela\n", encoding="utf-8")
    calls = []

    def fake_translate(texts, src, tgt, model=None, batch_size=16, sample=False, **kw):
        assert (src, tgt) == ("gn", "es")
        calls.append((len(texts), sample, batch_size))
        return [t.upper() for t in texts]

    monkeypatch.setattr(nmt, "translate", fake_translate)
    return tmp_path, calls


def test_writes_named_run_without_touching_e2_file(ddir):
    tmp, calls = ddir
    out = bt.backtranslate("bt2", n=25, chunk=10, lid=False, sample=True, batch_size=64)
    assert out == tmp / "train" / "bt2" / "bt2.jsonl"
    rows = sources.read_jsonl(out)
    assert [r["gn"] for r in rows] == [f"frase {i}" for i in range(25)]
    assert rows[3] == {"gn": "frase 3", "es": "FRASE 3", "source": "bt:madlad400"}
    assert calls == [(10, True, 64), (10, True, 64), (5, True, 64)]
    assert (tmp / "train" / "bt.jsonl").read_text(encoding="utf-8") == "sentinela\n"


def test_resumes_without_duplicates(ddir):
    tmp, calls = ddir
    bt.backtranslate("bt2", n=25, chunk=10, lid=False)
    parts = tmp / "train" / "bt2" / "parts"
    (parts / "00001.jsonl").unlink()
    (parts / "00002.jsonl").unlink()
    (parts / "00002.jsonl.tmp").write_text("{\"parcial", encoding="utf-8")
    calls.clear()
    rows = sources.read_jsonl(bt.backtranslate("bt2", n=25, chunk=10, lid=False))
    assert [c[0] for c in calls] == [10, 5]
    assert [r["gn"] for r in rows] == [f"frase {i}" for i in range(25)]


def test_changed_settings_refuse_to_mix(ddir):
    bt.backtranslate("bt2", n=25, chunk=10, lid=False)
    with pytest.raises(ValueError, match="bt2"):
        bt.backtranslate("bt2", n=25, chunk=5, lid=False)


def test_select_sentences_takes_all_wikipedia_then_fills_with_kuatia(monkeypatch):
    monkeypatch.setattr(bt, "eval_keys", lambda: set())
    monkeypatch.setattr(bt, "iter_wikipedia_docs", lambda: iter([
        ("wikipedia", "Ko ñe'ẽ ipuku porã ha iporã. Ñande ñe'ẽ oñeñe'ẽ Paraguái retãme."),
    ]))
    monkeypatch.setattr(bt, "iter_kuatia_docs", lambda: iter([
        ("madlad400", f"Ko'ã mba'e heta {i} jey oiko ko'ápe.") for i in range(10)
    ] + [("madlad400", "Ko ñe'ẽ ipuku porã ha iporã.")]))
    rows, stats = bt.select_sentences(n=5, lid=False)
    assert [r["source"] for r in rows] == ["wikipedia", "wikipedia", "madlad400", "madlad400", "madlad400"]
    assert stats["n"] == 5 and stats["lid_threshold"] is None
    assert stats["counts"]["madlad400:aceptada"] == 10 and stats["counts"]["madlad400:duplicada"] == 1


def test_select_sentences_with_fewer_candidates_than_n(monkeypatch):
    monkeypatch.setattr(bt, "eval_keys", lambda: set())
    monkeypatch.setattr(bt, "iter_wikipedia_docs", lambda: iter([("wikipedia", "Ko ñe'ẽ ipuku porã ha iporã.")]))
    monkeypatch.setattr(bt, "iter_kuatia_docs", lambda: iter([]))
    rows, stats = bt.select_sentences(n=500, lid=False)
    assert len(rows) == 1 and stats["n"] == 1
```

- [ ] **Step 3: Run and verify they fail**

Run: `uv run pytest tests/test_nmt.py tests/test_backtranslate.py -q`
Expected: FAIL (`translate() got an unexpected keyword argument 'sample'`, plus import/signature errors in `backtranslate`)

- [ ] **Step 4: Modify `src/guanani/nmt.py`**

Change the signature and body of `translate` and `_generate`:

```python
def translate(
    texts: list[str],
    src: str,
    tgt: str,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 16,
    sample: bool = False,
) -> list[str]:
    device = pick_device(device)
    tok, m = _load(model, device)
    tok.src_lang = CODES[src]
    if src == "gn":
        texts = [normalize(t) for t in texts]
    bos = tok.convert_tokens_to_ids(CODES[tgt])
    # Muestreo top-k para retrotraducción (mejor que beam a gran escala); beam 4 para evaluar.
    gen = {"do_sample": True, "top_k": 10, "num_beams": 1} if sample else {"num_beams": 4}
    if sample:
        torch.manual_seed(0)
    out = []
    for i in range(0, len(texts), batch_size):
        out += _generate(tok, m, texts[i : i + batch_size], bos, device, gen)
    return [normalize(o) for o in out] if tgt == "gn" else out


def _generate(tok, m, texts: list[str], bos: int, device: str, gen: dict) -> list[str]:
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    oom = False
    try:
        with torch.inference_mode():
            out = m.generate(**enc, forced_bos_token_id=bos, max_new_tokens=256, **gen)
    except torch.OutOfMemoryError:
        if len(texts) == 1:
            raise
        oom = True
    if not oom:
        return tok.batch_decode(out, skip_special_tokens=True)
    # Lotes con frases largas no caben en GPUs compartidas: se parte el lote en dos.
    # Fuera del except, para que el traceback no retenga los tensores del intento fallido.
    del enc
    torch.cuda.empty_cache()
    half = len(texts) // 2
    return _generate(tok, m, texts[:half], bos, device, gen) + _generate(tok, m, texts[half:], bos, device, gen)
```

- [ ] **Step 5: Rewrite `src/guanani/train/backtranslate.py`**

```python
"""Retrotraducción gn→es a data/train/<name>/, por bloques y reanudable."""

import json
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
) -> Path:
    d = train_dir() / name
    meta = {"n": n, "model": model, "sample": sample, "chunk": chunk, "seed": seed, "lid": lid}
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
        rows, stats = select_sentences(n, seed, lid)
        (d / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp = d / "source.jsonl.tmp"
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
```

- [ ] **Step 6: Adjust the CLI call** (so the suite does not break before Task 5)

In `src/guanani/cli.py`, in the `backtranslate` parser, replace the arguments with:

```python
    b.add_argument("--name", required=True, help="carpeta en data/train/ (p. ej. bt2)")
    b.add_argument("--n", type=int, default=500000)
    b.add_argument("--model")
    b.add_argument("--sample", action="store_true", help="muestreo top-k en vez de beam")
    b.add_argument("--batch-size", type=int, default=16)
    b.add_argument("--no-lid", action="store_true", help="sin filtro GlotLID")
```

and in the handler:

```python
                print(backtranslate(a.name, n=a.n, sample=a.sample, batch_size=a.batch_size,
                                    lid=not a.no_lid, **_kw(a)))
```

- [ ] **Step 7: Run and verify it passes**

Run: `uv run pytest -q`
Expected: all green (including 5 tests in `test_backtranslate.py` and 3 in `test_nmt.py`).

- [ ] **Step 8: Commit**

```bash
git add src/guanani/nmt.py src/guanani/train/backtranslate.py src/guanani/cli.py tests/test_nmt.py tests/test_backtranslate.py
git commit -m "feat: traducción por muestreo y retrotraducción con nombre, GlotLID y Wikipedia

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `btN` training data, upsampling and tag

**Files:**
- Modify: `src/guanani/train/lora.py`, `src/guanani/cli.py`
- Modify: `tests/test_lora.py`, `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `corpus.build_examples(..., bt_tag)`, `sources.read_jsonl`, `sources.train_dir`, `data._require`
- Produces:
  - `lora.data_file(kind: str) -> tuple[Path, str]` (`parallel`, `bt`, `btN`; anything else → a `ValueError` that mentions `btN`)
  - `lora.load_examples(data_kinds: list[str], upsample_parallel: int = 1, bt_tag: bool = False) -> tuple[list[dict], dict]` (counts `n_parallel`, `n_bt`, `n_examples`)
  - `lora.run(name, data_kinds, dev_size=300, models_dir=Path("models"), upsample_parallel=1, bt_tag=False, **train_kwargs) -> Path`
  - CLI `train lora --upsample-parallel N --bt-tag`
- Removes: `lora.DATA_FILES`

- [ ] **Step 1: Write the failing tests**

In `tests/test_lora.py`, add `from guanani.train import sources` to the imports, and append:

```python
def test_load_examples_upsamples_and_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "parallel.jsonl", [{"gn": "che", "es": "yo", "source": "s"}])
    sources.write_jsonl(tmp_path / "train" / "bt2" / "bt2.jsonl", [{"gn": "ore", "es": "nosotros", "source": "bt:x"}])
    ex, counts = lora.load_examples(["parallel", "bt2"], upsample_parallel=3, bt_tag=True)
    assert counts == {"n_parallel": 3, "n_bt": 1, "n_examples": 7}
    assert ex[0]["src"] == "che" and ex[-1]["src"] == "[BT] nosotros"
    assert sum(e["src"].startswith("[BT]") for e in ex) == 1


def test_data_file_resolves_named_bt_and_rejects_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    assert lora.data_file("bt2")[0] == tmp_path / "train" / "bt2" / "bt2.jsonl"
    assert lora.data_file("bt")[0] == tmp_path / "train" / "bt.jsonl"
    with pytest.raises(ValueError, match="btN"):
        lora.data_file("foo")
```

In `tests/test_cli.py`, add:

```python
def test_train_lora_missing_bt2_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "parallel.jsonl").write_text('{"gn": "a", "es": "b", "source": "s"}\n', encoding="utf-8")
    assert main(["train", "lora", "--name", "x", "--data", "parallel,bt2", "--upsample-parallel", "4", "--bt-tag"]) == 2
    assert "backtranslate" in capsys.readouterr().err
```

- [ ] **Step 2: Run and verify they fail**

Run: `uv run pytest tests/test_lora.py tests/test_cli.py -q`
Expected: FAIL (`module 'guanani.train.lora' has no attribute 'load_examples'`; in the CLI, `unrecognized arguments: --upsample-parallel`)

- [ ] **Step 3: Modify `src/guanani/train/lora.py`**

1. Add `import re` to the imports and delete the `DATA_FILES = {…}` line. After `CODES`, add:

```python
_BT_KIND = re.compile(r"bt\d+")
```

2. Before `run`, add:

```python
def data_file(kind: str) -> tuple[Path, str]:
    if kind == "parallel":
        return train_dir() / "parallel.jsonl", "train"
    if kind == "bt":
        return train_dir() / "bt.jsonl", "bt"
    if _BT_KIND.fullmatch(kind):
        return train_dir() / kind / f"{kind}.jsonl", "bt"
    raise ValueError(f"--data desconocido: {kind}; opciones: parallel, bt, btN (p. ej. bt2)")


def load_examples(data_kinds: list[str], upsample_parallel: int = 1, bt_tag: bool = False) -> tuple[list[dict], dict]:
    paths = {k: data._require(*data_file(k)) for k in data_kinds}
    parallel = read_jsonl(paths["parallel"]) * upsample_parallel if "parallel" in paths else []
    bt = [r for k, p in paths.items() if k != "parallel" for r in read_jsonl(p)]
    examples = build_examples(parallel, bt, bt_tag=bt_tag)
    return examples, {"n_parallel": len(parallel), "n_bt": len(bt), "n_examples": len(examples)}
```

3. Replace `run` with:

```python
def run(
    name: str,
    data_kinds: list[str],
    dev_size: int = 300,
    models_dir: Path = Path("models"),
    upsample_parallel: int = 1,
    bt_tag: bool = False,
    **train_kwargs,
) -> Path:
    examples, counts = load_examples(data_kinds, upsample_parallel, bt_tag)
    dev = data.load_flores("dev")[:dev_size]
    base = train_kwargs.get("base", BASE_MODEL)
    res = train(name, examples, dev, models_dir=models_dir, **train_kwargs)
    out = models_dir / name
    merge(base, out / "best_adapter", out)
    meta = {"name": name, "base": base, "data": data_kinds, "upsample_parallel": upsample_parallel,
            "bt_tag": bt_tag, "dev_size": dev_size, "train_kwargs": train_kwargs} | counts | res
    (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
```

- [ ] **Step 4: Modify `src/guanani/cli.py`**

In the `lora` parser, add:

```python
    lo.add_argument("--upsample-parallel", type=int, default=1, help="repite los pares reales N veces")
    lo.add_argument("--bt-tag", action="store_true", help="antepone [BT] al español retrotraducido")
```

and in the call to `run`, add `upsample_parallel=a.upsample_parallel, bt_tag=a.bt_tag,` as keyword arguments.

- [ ] **Step 5: Run and verify it passes**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/guanani/train/lora.py src/guanani/cli.py tests/test_lora.py tests/test_cli.py
git commit -m "feat: datos btN, sobremuestreo de pares reales y etiqueta [BT] en el entrenamiento

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Generate `bt2` on the server

**Files:** none versioned (the data lives in `~/Guanani/data/train/bt2/` on the server).

- [ ] **Step 1: Sync code and data**

Run:
```bash
rsync -az --exclude .venv --exclude models --exclude data --exclude .superpowers --exclude 'results/audio' \
  --exclude __pycache__ --exclude 'models-*.log' ./ user@gpu-server:Guanani/
rsync -az data/kuatia data/wikipedia user@gpu-server:Guanani/data/
rsync -azR data/./common_voice/test.tsv user@gpu-server:Guanani/data/
ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && UV_HTTP_TIMEOUT=300 uv sync -q && uv run pytest -q 2>&1 | tail -1 && ls data'
```
Expected: green suite on the server; `data` contains `common_voice`, `flores`, `jojajovai`, `kuatia`, `train`, `wikipedia`.

- [ ] **Step 2: Launch the back-translation**

Run:
```bash
ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run guanani train backtranslate --name bt2 --model models/e2 --sample --batch-size 64 > bt2.log 2>&1 < /dev/null &'
```
Expected: within a few minutes (GlotLID 1.7 GB download + selection), `data/train/bt2/stats.json` and `source.jsonl` appear.

- [ ] **Step 3: Review the selection**

Run:
```bash
ssh user@gpu-server 'cd ~/Guanani/data/train/bt2 && python3 -c "
import json; s=json.load(open(\"stats.json\")); print(s[\"n\"], s[\"lid_threshold\"])
from collections import Counter; c=Counter()
for k,v in s[\"counts\"].items(): c[k.split(\":\",1)[1] if k.count(\":\")==1 else k.split(\":\")[1]+\":\"+k.split(\":\")[2]]+=v
print(c.most_common(12))" && shuf -n 12 --random-source=<(yes) source.jsonl | cut -c1-140'
```
Expected:
- `n` close to 500,000 (if there are fewer candidates, note it);
- a threshold between 0.5 and 1;
- rejections such as `lid:gun_Latn`, `lid:spa_Latn`, etc.;
- a sample that is mostly Paraguayan Guarani.

Record the threshold, `n` and the main rejections in the ledger.

- [ ] **Step 4: Wait for it to finish and check**

Run: `ssh user@gpu-server 'cd ~/Guanani && wc -l data/train/bt2/bt2.jsonl && tail -2 bt2.log && shuf -n 5 --random-source=<(yes) data/train/bt2/bt2.jsonl | cut -c1-240'`
Expected: as many lines as `n`; pairs with real `gn`, synthetic `es` and `source` `bt:<source>`.

---

### Task 7: E3, E4, evaluation and conclusions

**Files:**
- Modify: `results/SUMMARY.md` (and `results/*.json` brought back from the server)

- [ ] **Step 1: Launch E3 and E4 back to back with their evaluations**

Run:
```bash
cat > /tmp/run_e3_e4.sh <<'EOF'
cd ~/Guanani
export PATH=$HOME/.local/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
common="--data parallel,bt2 --upsample-parallel 4 --epochs 1 --eval-steps 2000 --batch-size 16 --grad-accum 2"
uv run guanani train lora --name e3 $common > models-e3.log 2>&1
uv run guanani train lora --name e4 $common --bt-tag > models-e4.log 2>&1
for m in e3 e4; do
  for d in gn-es es-gn; do uv run guanani eval nmt flores --direction $d --model models/$m; done
  uv run guanani eval nmt jojajovai-anotado --direction gn-es --model models/$m
  for d in gn-es es-gn; do uv run guanani eval nmt jojajovai --direction $d --model models/$m; done
done
echo E3E4DONE
EOF
scp -q /tmp/run_e3_e4.sh user@gpu-server:Guanani/run_e3_e4.sh
ssh user@gpu-server 'cd ~/Guanani && setsid nohup bash run_e3_e4.sh > run_e3_e4.log 2>&1 < /dev/null &'
```
Expected: `models-e3.log` advances at ~1.2 steps/s; ~21,000 steps per model (~4.8 h each). On OOM, relaunch that model with `--batch-size 8 --grad-accum 4` and record a `Ruling:`.

- [ ] **Step 2: Bring back the results**

Run:
```bash
ssh user@gpu-server 'cd ~/Guanani && tail -1 run_e3_e4.log && for m in e3 e4; do python3 -c "import json; d=json.load(open(\"models/$m/train_meta.json\")); print(\"$m\", d[\"best_step\"], d[\"best_dev\"], d[\"n_examples\"])"; done'
rsync -az --exclude audio user@gpu-server:Guanani/results/ results/
grep -E "models/e[34]" results/SUMMARY.md | cut -d'|' -f4-9
```
Expected: `E3E4DONE`; 10 new rows (5 per model).

- [ ] **Step 3: Write conclusions**

In `results/SUMMARY.md`, before `## Conclusions (subproject 1, …)`, insert a section `## Conclusions (subproject 1b, <date>)` with:
- A table of base / E1 / E2 / E3 / E4 × (FLORES es→gn chrF++ and BLEU, FLORES gn→es chrF++ and BLEU, `jojajovai-anotado` gn→es chrF++), with ✅/❌ per threshold.
- The composition of `bt2`: `n`, the GlotLID threshold and the main rejections by label, from `data/train/bt2/stats.json` on the server.
- The effect of the tag (E4 − E3) and of scaling and filtering the back-translation (E3 − E2), per metric. Say explicitly whether chrF++ and BLEU agree.
- The duration of each phase on the RTX PRO 5000.
- The recommended next step.

- [ ] **Step 4: Commit**

```bash
git add results/
git commit -m "docs: resultados E3/E4 y conclusiones del subproyecto 1b

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
