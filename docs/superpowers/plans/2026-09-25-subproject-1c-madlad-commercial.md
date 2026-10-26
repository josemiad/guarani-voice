# Subproject 1c — Commercially licensed MADLAD-400: Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** All of it uses licence-approved data:
- support MADLAD-400 and full fine-tuning in the pipeline;
- audit the data licences;
- screen MADLAD-3B/7B;
- regenerate the back-translation with MADLAD (`bt3`);
- train E5 (full fine-tuning of MADLAD-3B) and, if warranted, E6 (LoRA on MADLAD-7B).

**Architecture:**
- `nmt.py` detects the model family from `config.model_type`: `t5` → MADLAD, with the `<2xx> ` prefix and bf16; otherwise NLLB as before.
- `lora.py` uses that family when tokenising and evaluating on dev. It gains `full` mode (Adafactor, no LoRA, saves `best_model/`) and a filter by approved sources.
- `sources.allowed_sources` reads the table in `docs/data-licenses.md`.
- `backtranslate` gains `--from` to reuse a sample.
- The `train lora` subcommand becomes `train nmt`, with `lora` as an alias.

The heavy runs go to `user@gpu-server`.

**Tech Stack:** Python 3.12, uv, torch, transformers 4.57 (T5ForConditionalGeneration, Seq2SeqTrainer, Adafactor), peft, sacrebleu, pytest, rsync/ssh.

**Spec:** `docs/superpowers/specs/2026-09-25-subproject-1c-madlad-commercial-design.md`

## Global Constraints

- **Candidate models:** `google/madlad400-3b-mt`, `google/madlad400-7b-mt` (Apache 2.0). NLLB only as a reference (CC-BY-NC).
- **MADLAD:** target prefix `<2gn> ` / `<2es> ` on the input text; no `src_lang` or `forced_bos_token_id`; bf16 on GPU.
- **Absolute thresholds** (FLORES+ devtest except the third): es→gn ≥ 36.05; gn→es ≥ 39.85; `jojajovai-anotado` gn→es ≥ 49.31. Report chrF++, BLEU and licence.
- **Licence rule:** a non-commercial or unknown source → excluded from commercial training. The table in `docs/data-licenses.md` has the columns `source` and `Commercial`; only `yes` approves.
- **`bt3`:** the same sample as `bt2` (`--from bt2`), back-translated with `google/madlad400-3b-mt`, `--sample`, top-k 10.
- **E5:** `--base google/madlad400-3b-mt --full --data parallel,bt3 --upsample-parallel 4 --epochs 1 --lr 1e-4 --sources-allow docs/data-licenses.md`. Batch size and evaluation steps are set after the memory test.
- **E6:** only if, in the screening, MADLAD-7B beats MADLAD-3B by ≥ 2 chrF++ on es→gn (FLORES devtest). LoRA r=16 in bf16; if it does not fit in 24 GB, it is dropped (no quantisation).
- **Full fine-tuning** is pure bf16 with Adafactor, because the fp32 weights of 3B plus gradients do not fit in 24 GB. The memory test checks that the loss goes down.
- **Server:** `user@gpu-server`, `~/Guanani`, `uv` in `~/.local/bin`. Long jobs run with `setsid nohup … < /dev/null &`; results come back with `rsync results/`.
- `pkill` with patterns that do not match the command itself.
- Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A locally saved fine-tuned model (`models/e5`) must be detected as MADLAD from its `config.json`, not from its name (Task 1).
2. With `--sources-allow`, a data type that is empty after filtering → an error before loading any model, never training on zero examples (Task 3).
3. `--from bt2` must not modify anything in `data/train/bt2/` (Task 3).
4. In `--full`, `best_model/` always holds the best checkpoint, and at the end `models/<name>/` holds a complete loadable model with no adapter (Task 2).
5. `train lora …` keeps working exactly as before (Task 2).

---

### Task 1: MADLAD support in `nmt`

**Files:**
- Modify: `src/guanani/nmt.py`
- Modify: `tests/test_nmt.py` (append)

**Interfaces:**
- Produces:
  - `MADLAD_PREFIX = {"gn": "<2gn> ", "es": "<2es> "}`
  - `family_of(config) -> str` (`"madlad"` if `config.model_type == "t5"`, otherwise `"nllb"`; accepts `None`)
  - `load_seq2seq(model, dtype)` — now also works with models that lack safetensors (catches `OSError` and loads with `use_safetensors=False`)
  - `translate(...)` with the same signature; with MADLAD it prepends the target-language prefix and does not pass `forced_bos_token_id`
  - `_generate(tok, m, texts, device, gen)` — `forced_bos_token_id` goes inside `gen`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_nmt.py`)

```python
from types import SimpleNamespace


class _T5Model:
    config = SimpleNamespace(model_type="t5")

    def __init__(self):
        self.kwargs = []

    def generate(self, texts, **kw):
        self.kwargs.append(kw)
        return texts


def test_family_of_detects_t5_as_madlad():
    assert nmt.family_of(SimpleNamespace(model_type="t5")) == "madlad"
    assert nmt.family_of(SimpleNamespace(model_type="m2m_100")) == "nllb"
    assert nmt.family_of(None) == "nllb"


def test_madlad_uses_target_prefix_and_no_forced_bos(monkeypatch):
    m = _T5Model()
    monkeypatch.setattr(nmt, "_load", lambda model, device: (_Tok(), m))
    out = nmt.translate(["ko ñe'ẽ"], "gn", "es", device="cpu")
    assert out == ["<2ES> KO ÑE'Ẽ"]
    assert "forced_bos_token_id" not in m.kwargs[0]


def test_nllb_still_forces_bos(monkeypatch):
    m = _RecordsKwargs()
    monkeypatch.setattr(nmt, "_load", lambda model, device: (_Tok(), m))
    nmt.translate(["a"], "es", "gn", device="cpu")
    assert m.kwargs[0]["forced_bos_token_id"] == 0
```

- [ ] **Step 2: Run and verify it fails**

Run: `uv run pytest tests/test_nmt.py -q`
Expected: FAIL with `AttributeError: module 'guanani.nmt' has no attribute 'family_of'`

- [ ] **Step 3: Implement in `src/guanani/nmt.py`**

1. Change the transformers import to `from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer`.
2. After `CODES`, add: `MADLAD_PREFIX = {"gn": "<2gn> ", "es": "<2es> "}`.
3. Replace `load_seq2seq` and `_load` with:

```python
def family_of(config) -> str:
    return "madlad" if getattr(config, "model_type", None) == "t5" else "nllb"


def load_seq2seq(model: str, dtype: torch.dtype):
    # Safetensors primero: evita volver a descargar pytorch_model.bin si solo está en caché la conversión.
    # Si no hay safetensors, o la conversión omite pesos compartidos (NLLB-600M), se usa el .bin.
    try:
        m = AutoModelForSeq2SeqLM.from_pretrained(model, dtype=dtype, use_safetensors=True)
    except OSError:
        return AutoModelForSeq2SeqLM.from_pretrained(model, dtype=dtype, use_safetensors=False)
    if any(p.is_meta for p in m.parameters()):
        m = AutoModelForSeq2SeqLM.from_pretrained(model, dtype=dtype, use_safetensors=False)
    return m


@cache
def _load(model: str, device: str):
    tok = AutoTokenizer.from_pretrained(model)
    # T5 (MADLAD) desborda en fp16: bf16 en GPU.
    half = torch.bfloat16 if family_of(AutoConfig.from_pretrained(model)) == "madlad" else torch.float16
    return tok, load_seq2seq(model, half if device == "cuda" else torch.float32).to(device).eval()
```

4. Replace the body of `translate` (from `device = pick_device(device)`) and `_generate` with:

```python
    device = pick_device(device)
    tok, m = _load(model, device)
    if src == "gn":
        texts = [normalize(t) for t in texts]
    # Muestreo top-k para retrotraducción (mejor que beam a gran escala); beam 4 para evaluar.
    gen = {"do_sample": True, "top_k": 10, "num_beams": 1} if sample else {"num_beams": 4}
    if family_of(getattr(m, "config", None)) == "madlad":
        texts = [MADLAD_PREFIX[tgt] + t for t in texts]
    else:
        tok.src_lang = CODES[src]
        gen["forced_bos_token_id"] = tok.convert_tokens_to_ids(CODES[tgt])
    if sample:
        torch.manual_seed(0)
    out = []
    for i in range(0, len(texts), batch_size):
        out += _generate(tok, m, texts[i : i + batch_size], device, gen)
    return [normalize(o) for o in out] if tgt == "gn" else out


def _generate(tok, m, texts: list[str], device: str, gen: dict) -> list[str]:
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    oom = False
    try:
        with torch.inference_mode():
            out = m.generate(**enc, max_new_tokens=256, **gen)
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
    return _generate(tok, m, texts[:half], device, gen) + _generate(tok, m, texts[half:], device, gen)
```

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest -q`
Expected: all green (including the 3 new tests).

- [ ] **Step 5: Commit**

```bash
git add src/guanani/nmt.py tests/test_nmt.py
git commit -m "feat: soporte de MADLAD-400 en la traducción (prefijo <2xx>, bf16)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Training with MADLAD and full fine-tuning; `train nmt`

**Files:**
- Modify: `src/guanani/train/lora.py`, `src/guanani/cli.py`
- Modify: `tests/test_lora.py`, `tests/test_cli.py`, `tests/test_models_slow.py` (append)

**Interfaces:**
- Consumes: `nmt.MADLAD_PREFIX`, `nmt.family_of`, `nmt.load_seq2seq`
- Produces:
  - `tokenize_examples(tok, examples, max_len=MAX_LEN, family="nllb")`
  - `dev_chrf(model, tok, pairs, direction, batch_size=16)` — detects the family with `family_of(model.config)`
  - `DevChrF(tok, dev_pairs, eval_steps, out_dir, best_name="best_adapter")`
  - `train(..., full: bool = False, ...)` — with `full=True`: no LoRA, `optim="adafactor"`, best model in `out/best_model/`
  - `finalize_full(out_dir: Path, base: str) -> None` — moves `best_model/*` to `out_dir/` and saves the tokenizer
  - `prepare_out_dir` also deletes `best_model`
  - CLI `guanani train nmt …` (alias `lora`) with `--full`

- [ ] **Step 1: Write the failing tests**

In `tests/test_lora.py`, add:

```python
class _RecTok:
    def __init__(self):
        self.calls = []

    def __call__(self, srcs, text_target, max_length, truncation):
        self.calls += srcs
        n = len(srcs)
        return {"input_ids": [[1]] * n, "attention_mask": [[1]] * n, "labels": [[2]] * n}


def test_tokenize_examples_madlad_prefixes_target_language():
    tok = _RecTok()
    ex = [
        {"src": "che", "tgt": "yo", "src_lang": "grn_Latn", "tgt_lang": "spa_Latn"},
        {"src": "yo", "tgt": "che", "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
    ]
    out = lora.tokenize_examples(tok, ex, family="madlad")
    assert sorted(tok.calls) == ["<2es> che", "<2gn> yo"]
    assert len(out) == 2


def test_finalize_full_moves_best_model_and_saves_tokenizer(tmp_path, monkeypatch):
    best = tmp_path / "best_model"
    best.mkdir()
    (best / "config.json").write_text("{}", encoding="utf-8")
    (best / "model.safetensors").write_text("w", encoding="utf-8")

    class _FakeTok:
        def save_pretrained(self, d):
            (Path(d) / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(lora.AutoTokenizer, "from_pretrained", lambda base: _FakeTok())
    lora.finalize_full(tmp_path, "base")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "model.safetensors", "tokenizer.json"]


def test_fresh_run_also_discards_best_model(tmp_path):
    (tmp_path / "best_model").mkdir()
    lora.prepare_out_dir(tmp_path, resume=False)
    assert not (tmp_path / "best_model").exists()
```

and `from pathlib import Path` to that file's imports.

In `tests/test_cli.py`, add:

```python
def test_train_nmt_is_the_new_name_and_lora_still_works(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "parallel.jsonl").write_text('{"gn": "a", "es": "b", "source": "s"}\n', encoding="utf-8")
    for cmd in ("nmt", "lora"):
        assert main(["train", cmd, "--name", "x", "--data", "parallel,bt3", "--full"]) == 2
        assert "backtranslate" in capsys.readouterr().err
```

In `tests/test_models_slow.py`, add:

```python
def test_full_finetune_tiny_t5_saves_complete_model(tmp_path):
    from guanani.nmt import translate
    from guanani.train.corpus import build_examples
    from guanani.train.lora import finalize_full, train

    pairs = [{"gn": f"Che réra Juan {i}.", "es": f"Me llamo Juan {i}.", "source": "t"} for i in range(20)]
    dev = [{"gn": "Mba'éichapa reiko?", "es": "¿Cómo estás?"}] * 2
    base = "hf-internal-testing/tiny-random-t5"
    res = train("tinyfull", build_examples(pairs), dev, base=base, full=True, batch_size=4, grad_accum=1,
                max_steps=4, eval_steps=2, models_dir=tmp_path)
    out = tmp_path / "tinyfull"
    assert res["best_step"] in (2, 4) and (out / "best_model").exists()
    finalize_full(out, base)
    assert not (out / "best_model").exists() and not (out / "adapter_config.json").exists()
    assert isinstance(translate(["Mba'éichapa"], "gn", "es", model=str(out))[0], str)
```

- [ ] **Step 2: Run and verify they fail**

Run: `uv run pytest tests/test_lora.py tests/test_cli.py -q`
Expected: FAIL (`tokenize_examples() got an unexpected keyword argument 'family'`, `no attribute 'finalize_full'`, `invalid choice: 'nmt'`)

- [ ] **Step 3: Modify `src/guanani/train/lora.py`**

1. Import: replace `from guanani.nmt import load_seq2seq` with `from guanani.nmt import MADLAD_PREFIX, family_of, load_seq2seq`. After `CODES`, add: `_MADLAD_CODE = {"grn_Latn": "gn", "spa_Latn": "es"}`.

2. Replace `tokenize_examples` with:

```python
def tokenize_examples(tok, examples: list[dict], max_len: int = MAX_LEN, family: str = "nllb") -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for ex in examples:
        groups.setdefault((ex["src_lang"], ex["tgt_lang"]), []).append(ex)
    out = []
    for (src_lang, tgt_lang), exs in groups.items():
        srcs = [e["src"] for e in exs]
        if family == "madlad":
            srcs = [MADLAD_PREFIX[_MADLAD_CODE[tgt_lang]] + s for s in srcs]
        else:
            tok.src_lang, tok.tgt_lang = src_lang, tgt_lang
        enc = tok(srcs, text_target=[e["tgt"] for e in exs], max_length=max_len, truncation=True)
        out += [
            {"input_ids": i, "attention_mask": a, "labels": lab}
            for i, a, lab in zip(enc["input_ids"], enc["attention_mask"], enc["labels"])
        ]
    return out
```

3. Replace `dev_chrf` with:

```python
def dev_chrf(model, tok, pairs: list[dict], direction: str, batch_size: int = 16) -> float:
    src, tgt = direction.split("-")
    srcs = [p[src] for p in pairs]
    gen = {"num_beams": 1}
    if family_of(model.config) == "madlad":
        srcs = [MADLAD_PREFIX[tgt] + s for s in srcs]
    else:
        tok.src_lang = CODES[src]
        gen["forced_bos_token_id"] = tok.convert_tokens_to_ids(CODES[tgt])
    was_training = model.training
    model.eval()
    hyps = []
    for i in range(0, len(srcs), batch_size):
        enc = tok(srcs[i : i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_LEN, **gen)
        hyps += tok.batch_decode(out, skip_special_tokens=True)
    if was_training:
        model.train()
    refs = [p[tgt] for p in pairs]
    if tgt == "gn":
        hyps, refs = [normalize(h) for h in hyps], [normalize(r) for r in refs]
    return score_mt(hyps, refs)["chrf++"]
```

4. In `DevChrF.__init__`, add the parameter `best_name: str = "best_adapter"` and store `self.best_dir = out_dir / best_name`. In `_evaluate`, change `model.save_pretrained(self.out_dir / "best_adapter")` to `model.save_pretrained(self.best_dir)`.

5. In `prepare_out_dir`, change the tuple to `("best_adapter", "best_model", "checkpoints")`.

6. In `train`, make three changes:
   - Add the parameter `full: bool = False` (after `resume`).
   - Replace everything from `model = load_seq2seq(...)` to `cb = DevChrF(...)` with:

```python
    model = load_seq2seq(base, torch.bfloat16)
    family = family_of(model.config)
    if not full:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 target_modules=TARGET_MODULES, task_type="SEQ_2_SEQ_LM"))
        model.print_trainable_parameters()
    cb = DevChrF(tok, dev_pairs, eval_steps, out, best_name="best_model" if full else "best_adapter")
```

   - Add `optim="adafactor" if full else "adamw_torch",` to `Seq2SeqTrainingArguments`, and change `train_dataset=tokenize_examples(tok, examples)` to `train_dataset=tokenize_examples(tok, examples, family=family)`.

7. After `merge`, add:

```python
def finalize_full(out_dir: Path, base: str) -> None:
    best = out_dir / "best_model"
    for f in best.iterdir():
        shutil.move(str(f), out_dir / f.name)
    best.rmdir()
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
```

8. In `run`, replace `merge(base, out / "best_adapter", out)` with:

```python
    if train_kwargs.get("full"):
        finalize_full(out, base)
    else:
        merge(base, out / "best_adapter", out)
```

- [ ] **Step 4: Modify `src/guanani/cli.py`**

Change `lo = tsub.add_parser("lora", help="entrena LoRA, fusiona y guarda en models/<name>")` to:

```python
    lo = tsub.add_parser("nmt", aliases=["lora"], help="entrena NLLB/MADLAD (LoRA o --full) y guarda en models/<name>")
```

Add `lo.add_argument("--full", action="store_true", help="fine-tuning completo (Adafactor) en vez de LoRA")`, and add `full=a.full,` to the call to `run`.

- [ ] **Step 5: Run and verify it passes**

Run: `uv run pytest -q && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run pytest -q -m slow tests/test_models_slow.py::test_full_finetune_tiny_t5_saves_complete_model tests/test_models_slow.py::test_lora_tiny_training_merges_into_translatable_model`
Expected: fast suite green; the 2 slow tests pass. The tiny T5 has no safetensors, so it exercises the `.bin` fallback from Task 1. If the local GPU is busy with the user's services and runs out of memory, run the slow tests on the server after syncing.

- [ ] **Step 6: Commit**

```bash
git add src/guanani/train/lora.py src/guanani/cli.py tests/test_lora.py tests/test_cli.py tests/test_models_slow.py
git commit -m "feat: entrenamiento con MADLAD y fine-tuning completo; guanani train nmt

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Licence filter and `backtranslate --from`

**Files:**
- Modify: `src/guanani/train/sources.py`, `src/guanani/train/lora.py`, `src/guanani/train/backtranslate.py`, `src/guanani/cli.py`
- Modify: `tests/test_sources.py`, `tests/test_lora.py`, `tests/test_backtranslate.py` (append)

**Interfaces:**
- Produces:
  - `sources.allowed_sources(path: Path) -> set[str]`:
    - reads the first Markdown table with the columns `source` and `comercial` (case-insensitive);
    - approves rows marked `sí`/`si`;
    - a `source` cell may hold several sources, comma-separated and in backticks;
    - raises `ValueError` if it finds no table.
  - `lora.load_examples(data_kinds, upsample_parallel=1, bt_tag=False, allow: set[str] | None = None)` — filters by `source` before upsampling; `ValueError` if any data type ends up empty
  - `lora.run(..., sources_allow: str | None = None, ...)` — saves `sources_allow` and the approved list in `train_meta.json`
  - `backtranslate.backtranslate(name, ..., from_: str | None = None)` — copies `data/train/<from_>/source.jsonl` (and `stats.json` if present) without modifying the source; `from` is recorded in `meta.json`
  - CLI: `train nmt --sources-allow PATH`; `train backtranslate --from NAME`

- [ ] **Step 1: Write the failing tests**

In `tests/test_sources.py`:

```python
def test_allowed_sources_reads_approved_rows(tmp_path):
    p = tmp_path / "lic.md"
    p.write_text(
        "# Licencias\n\n"
        "| source | Origen | Comercial | Notas |\n"
        "|---|---|---|---|\n"
        "| `jojajovai` | Jojajovai | sí | x |\n"
        "| `bt:madlad400`, `bt:fineweb-2` | KuatIA | Sí | y |\n"
        "| `bt:guarania-scraped` | KuatIA | desconocido | z |\n"
        "| `americasnlp` | AmericasNLP | no | w |\n",
        encoding="utf-8",
    )
    assert sources.allowed_sources(p) == {"jojajovai", "bt:madlad400", "bt:fineweb-2"}


def test_allowed_sources_without_table_raises(tmp_path):
    p = tmp_path / "lic.md"
    p.write_text("sin tabla\n", encoding="utf-8")
    with pytest.raises(ValueError, match="comercial"):
        sources.allowed_sources(p)
```

In `tests/test_lora.py`:

```python
def test_load_examples_filters_by_allowed_sources_before_upsampling(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "parallel.jsonl",
                        [{"gn": "che", "es": "yo", "source": "jojajovai"}, {"gn": "nde", "es": "tú", "source": "americasnlp"}])
    sources.write_jsonl(tmp_path / "train" / "bt3" / "bt3.jsonl",
                        [{"gn": "ore", "es": "nosotros", "source": "bt:wikipedia"}, {"gn": "pe", "es": "vosotros", "source": "bt:x"}])
    ex, counts = lora.load_examples(["parallel", "bt3"], upsample_parallel=2, allow={"jojajovai", "bt:wikipedia"})
    assert counts == {"n_parallel": 2, "n_bt": 1, "n_examples": 5}
    assert {e["src"] for e in ex} == {"che", "yo", "nosotros"}


def test_load_examples_empty_after_filter_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "parallel.jsonl", [{"gn": "che", "es": "yo", "source": "jojajovai"}])
    sources.write_jsonl(tmp_path / "train" / "bt3" / "bt3.jsonl", [{"gn": "ore", "es": "nosotros", "source": "bt:x"}])
    with pytest.raises(ValueError, match="bt3"):
        lora.load_examples(["parallel", "bt3"], allow={"jojajovai"})
```

In `tests/test_backtranslate.py`:

```python
def test_from_reuses_source_without_touching_origin(ddir):
    tmp, calls = ddir
    origin = tmp / "train" / "bt2" / "source.jsonl"
    before = origin.read_bytes()
    out = bt.backtranslate("bt3", from_="bt2", chunk=10, lid=False)
    assert origin.read_bytes() == before
    assert sorted(p.name for p in (tmp / "train" / "bt2").iterdir()) == ["source.jsonl"]
    assert [r["gn"] for r in sources.read_jsonl(out)] == [f"frase {i}" for i in range(25)]


def test_from_missing_origin_raises(ddir):
    from guanani import data

    with pytest.raises(data.DataMissing):
        bt.backtranslate("bt4", from_="bt9", lid=False)
```

- [ ] **Step 2: Run and verify they fail**

Run: `uv run pytest tests/test_sources.py tests/test_lora.py tests/test_backtranslate.py -q`
Expected: FAIL (`no attribute 'allowed_sources'`, `unexpected keyword argument 'allow'`, `unexpected keyword argument 'from_'`)

- [ ] **Step 3: Implement**

In `src/guanani/train/sources.py`, add:

```python
def allowed_sources(path: Path) -> set[str]:
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]
    header = [h.lower() for h in rows[0]] if rows else []
    if "source" not in header or "comercial" not in header:
        raise ValueError(f"{path}: falta una tabla con columnas 'source' y 'comercial'")
    si, ci = header.index("source"), header.index("comercial")
    allowed = set()
    for r in rows[1:]:
        if len(r) <= max(si, ci) or set(r[si]) <= set("-: "):
            continue
        if r[ci].lower() in ("sí", "si"):
            allowed |= {s.strip().strip("`") for s in r[si].split(",") if s.strip()}
    return allowed
```

In `src/guanani/train/lora.py`, replace `load_examples` with:

```python
def load_examples(
    data_kinds: list[str], upsample_parallel: int = 1, bt_tag: bool = False, allow: set[str] | None = None
) -> tuple[list[dict], dict]:
    paths = {k: data._require(*data_file(k)) for k in data_kinds}
    rows = {k: read_jsonl(p) for k, p in paths.items()}
    if allow is not None:
        rows = {k: [r for r in v if r["source"] in allow] for k, v in rows.items()}
        empty = [k for k, v in rows.items() if not v]
        if empty:
            raise ValueError(f"Sin ejemplos con licencia aprobada en: {empty}")
    parallel = rows.get("parallel", []) * upsample_parallel
    bt = [r for k, v in rows.items() if k != "parallel" for r in v]
    examples = build_examples(parallel, bt, bt_tag=bt_tag)
    return examples, {"n_parallel": len(parallel), "n_bt": len(bt), "n_examples": len(examples)}
```

Then in `run`:
- Add the parameter `sources_allow: str | None = None` (after `bt_tag`).
- Replace the first line of the body with:

```python
    allow = allowed_sources(Path(sources_allow)) if sources_allow else None
    examples, counts = load_examples(data_kinds, upsample_parallel, bt_tag, allow)
```

- Replace the current sources import with `from guanani.train.sources import allowed_sources, read_jsonl, train_dir`.
- Add `"sources_allow": sources_allow, "allowed_sources": sorted(allow) if allow else None,` to the `meta` dictionary.

In `src/guanani/train/backtranslate.py`:
- Add `import shutil`.
- Add the parameter `from_: str | None = None` (last).
- Add `"from": from_` to the `meta` dictionary.
- Replace the `if not src_path.exists():` block with:

```python
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
```

In `src/guanani/cli.py`:
- In the `backtranslate` parser, add `b.add_argument("--from", dest="from_", help="reutiliza la muestra de data/train/<from>/")` and pass `from_=a.from_` to `backtranslate`.
- In the `nmt` parser, add `lo.add_argument("--sources-allow", help="tabla de licencias: solo fuentes con Comercial = sí")` and pass `sources_allow=a.sources_allow` to `run`.

- [ ] **Step 4: Run and verify it passes**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/guanani/train/sources.py src/guanani/train/lora.py src/guanani/train/backtranslate.py src/guanani/cli.py tests/test_sources.py tests/test_lora.py tests/test_backtranslate.py
git commit -m "feat: filtro de datos por licencia y backtranslate --from

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Data licence audit

**Files:**
- Create: `docs/data-licenses.md`
- Test: `tests/test_sources.py` (add a test on the real file)

- [ ] **Step 1: Collect the licences**

Check each source's official page (with WebFetch) and note the licence, commercial use and conditions:

| `source` | Where to look |
|---|---|
| `jojajovai` | https://github.com/pln-fing-udelar/jojajovai (LICENSE and README) and the LREC 2022 paper |
| `americasnlp` | https://github.com/AmericasNLP/americasnlp2021 (README of `data/guarani-spanish`) |
| `bt:wikipedia` | https://huggingface.co/datasets/wikimedia/wikipedia |
| `bt:madlad400` | https://huggingface.co/datasets/allenai/MADLAD-400 |
| `bt:fineweb-2` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| `bt:hplt-3` | https://hplt-project.org/datasets |
| `bt:FinePDF` | https://huggingface.co/datasets/HuggingFaceFW/finepdfs |
| `bt:culturax` | https://huggingface.co/datasets/uonlp/CulturaX |
| `bt:multi-wiki-qa` | card at https://huggingface.co/datasets/guaran-ia/kuatia |
| `bt:guarania-scraped` | card at https://huggingface.co/datasets/guaran-ia/kuatia |
| (KuatIA as a whole) | https://huggingface.co/datasets/guaran-ia/kuatia (the compilation's licence) |

Rules for the `Commercial` column:
- `yes` only if the licence allows commercial use (e.g. CC0, CC BY, CC BY-SA, ODC-By, Apache);
- `no` if it is NC or forbids commercial use;
- `unknown` if there is no clear licence.

If the KuatIA compilation's licence is more restrictive than the original source's, the more restrictive one applies to the `bt:*` rows.

- [ ] **Step 2: Write `docs/data-licenses.md`**

Required structure (the first table is the one `allowed_sources` reads):

```markdown
# Licencias de datos y modelos

Análisis técnico, no asesoramiento legal. Revisado: <fecha>.

## Datos de entrenamiento

| source | Origen | Licencia | Comercial | Condiciones | Enlace |
|---|---|---|---|---|---|
| `jojajovai` | … | … | sí/no/desconocido | … | … |
| … una fila por cada `source` de la tabla del Step 1 … |

## Solo evaluación (no entran en el entrenamiento)

| Recurso | Licencia | Enlace |
|---|---|---|
| FLORES+ | … | … |
| Common Voice 17 (gn) | … | … |

## Modelos y herramientas

| Recurso | Licencia | Uso |
|---|---|---|
| MADLAD-400-3B/7B-MT | Apache 2.0 | candidato comercial |
| NLLB-200, MMS | CC-BY-NC-4.0 | solo referencia |
| GlotLID | Apache 2.0 | filtrado de datos |
| COMET wmt22-comet-da | … | evaluación |

## Consecuencias
<qué fuentes quedan fuera y cuánto pesan: contar filas por source en data/train/parallel.jsonl y data/train/bt2/source.jsonl>
```

For the consequences section, count with:
`uv run python -c "import json,collections; print(collections.Counter(json.loads(l)['source'] for l in open('data/train/parallel.jsonl')))"` (locally). Do the same on the server for `data/train/bt2/source.jsonl`, with `bt:` in front of the source name, since that is how it will appear in `bt3.jsonl`.

- [ ] **Step 3: Test on the real file** (append to `tests/test_sources.py`)

```python
def test_real_license_table_parses_and_uses_known_sources():
    from pathlib import Path

    allowed = sources.allowed_sources(Path("docs/data-licenses.md"))
    known = {"jojajovai", "americasnlp", "bt:wikipedia", "bt:madlad400", "bt:fineweb-2", "bt:hplt-3",
             "bt:FinePDF", "bt:culturax", "bt:multi-wiki-qa", "bt:guarania-scraped"}
    assert allowed <= known
```

Run: `uv run pytest tests/test_sources.py -q`
Expected: passes. If `allowed` ends up empty, or with no real-pair source at all, **stop and warn the user** before Task 7: E5 could not be trained on real pairs.

- [ ] **Step 4: Commit**

```bash
git add docs/data-licenses.md tests/test_sources.py
git commit -m "docs: auditoría de licencias de datos y modelos

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: MADLAD screening on the server

**Files:** new rows in `results/`

- [ ] **Step 1: Sync**

Run:
```bash
rsync -az --exclude .venv --exclude models --exclude data --exclude .superpowers --exclude 'results/audio' \
  --exclude __pycache__ --exclude 'models-*.log' ./ user@gpu-server:Guanani/
ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && UV_HTTP_TIMEOUT=300 uv sync -q && uv run pytest -q 2>&1 | tail -1'
```
Expected: green suite on the server.

- [ ] **Step 2: Run the screening**

Run:
```bash
ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && setsid nohup bash -c "for m in google/madlad400-3b-mt google/madlad400-7b-mt; do for d in gn-es es-gn; do UV_HTTP_TIMEOUT=300 uv run guanani eval nmt flores --direction \$d --model \$m 2>&1 | grep -E \"results/|Error\"; done; done; echo CRIBADONE" > criba.log 2>&1 < /dev/null &'
```
Expected: `criba.log` ends with `CRIBADONE` and 4 result paths (downloading the 7B, ~30 GB, may take a while).

- [ ] **Step 3: Bring back and decide on E6**

Run: `rsync -az --exclude audio user@gpu-server:Guanani/results/ results/ && grep -E "madlad400" results/SUMMARY.md | cut -d'|' -f4-8`
Expected: 4 rows. Note the chrF++/BLEU and the decision in the ledger: E6 runs if 7B es→gn ≥ 3B es→gn + 2.0.

- [ ] **Step 4: Commit**

```bash
git add results/
git commit -m "docs: criba de MADLAD-400 3B y 7B en FLORES devtest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `bt3` back-translation with MADLAD-3B

**Files:** none versioned (`~/Guanani/data/train/bt3/` on the server)

- [ ] **Step 1: Launch**

Run:
```bash
ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run guanani train backtranslate --name bt3 --from bt2 --model google/madlad400-3b-mt --sample --batch-size 64 > bt3.log 2>&1 < /dev/null &'
```
Expected: `data/train/bt3/source.jsonl` identical to `bt2`'s (check with `cmp`); blocks appearing in `bt3/parts/`.

- [ ] **Step 2: Check quality and pace after ~10 blocks**

Run: `ssh user@gpu-server 'cd ~/Guanani/data/train && cmp bt2/source.jsonl bt3/source.jsonl && echo misma-muestra; ls bt3/parts | wc -l; shuf -n 5 --random-source=<(yes) bt3/parts/00003.jsonl | cut -c1-240'`
Expected: `misma-muestra` (same sample); reasonable Spanish translations (not empty, not in Guarani). Estimate the total duration from the block rate and note it.

- [ ] **Step 3: Wait for the end**

Run: `ssh user@gpu-server 'wc -l ~/Guanani/data/train/bt3/bt3.jsonl'`
Expected: 499,420 lines.

---

### Task 7: E5 — full fine-tuning of MADLAD-3B

**Files:** new rows in `results/`

- [ ] **Step 1: Sync the licence table**

Run: `rsync -az docs/data-licenses.md user@gpu-server:Guanani/docs/`

- [ ] **Step 2: Memory and learning test**

Run:
```bash
ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run guanani train nmt --name probe5 --base google/madlad400-3b-mt --full --data parallel,bt3 --upsample-parallel 4 --lr 1e-4 --sources-allow docs/data-licenses.md --batch-size 8 --grad-accum 4 --max-steps 60 --eval-steps 30 --dev-size 50 > probe5.log 2>&1; tr "\r" "\n" < probe5.log | grep -E "loss|\[dev\]|s/it|it/s|Error|OutOfMemory" | tail -8; nvidia-smi --query-gpu=memory.used --format=csv,noheader; rm -rf models/probe5'
```
Expected: no OOM; the `loss` falls between the first and last log entries; two `[dev]` lines with reasonable chrF++ (not close to 0).

If something goes wrong:
- **OOM:** repeat with `--batch-size 4 --grad-accum 8`, then `2 × 16` if it persists, and record a `Ruling:`.
- **The loss does not fall, or dev collapses:** stop and warn the user (possibly a pure-bf16 problem).

Then plan the run:
- total steps = ceil(n_examples / 32), where n_examples comes from the `train_meta` log or from `load_examples`;
- estimate the duration;
- set `--eval-steps` for ~10 evaluations.

- [ ] **Step 3: Launch E5 with its evaluations**

Run (replacing `B`, `A` and `S` with the batch size, accumulation and evaluation steps from the test):
```bash
ssh user@gpu-server 'cd ~/Guanani && cat > run_e5.sh <<EOF
cd ~/Guanani
export PATH=\$HOME/.local/bin:\$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run guanani train nmt --name e5 --base google/madlad400-3b-mt --full --data parallel,bt3 --upsample-parallel 4 --epochs 1 --lr 1e-4 --sources-allow docs/data-licenses.md --batch-size B --grad-accum A --eval-steps S > models-e5.log 2>&1
echo "E5 exit=\$? \$(date +%H:%M)"
for d in gn-es es-gn; do uv run guanani eval nmt flores --direction \$d --model models/e5; done
uv run guanani eval nmt jojajovai-anotado --direction gn-es --model models/e5
for d in gn-es es-gn; do uv run guanani eval nmt jojajovai --direction \$d --model models/e5; done
echo E5DONE
EOF
setsid nohup bash run_e5.sh > run_e5.log 2>&1 < /dev/null &'
```
Expected: `run_e5.log` ends with `E5 exit=0` and `E5DONE`.

- [ ] **Step 4: Bring back the results**

Run: `rsync -az --exclude audio user@gpu-server:Guanani/results/ results/ && grep "models/e5" results/SUMMARY.md | cut -d'|' -f4-8`
Expected: 5 rows.

- [ ] **Step 5: Commit**

```bash
git add results/
git commit -m "docs: resultados E5 (fine-tuning completo de MADLAD-3B)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: E6 (conditional) and conclusions

**Files:**
- Modify: `results/SUMMARY.md`

- [ ] **Step 1: E6 only if the screening justifies it**

If Task 5 decided against E6, note the reason and go to Step 3.

If E6 runs:
1. Run a memory test as in Task 7 Step 2, but without `--full` (LoRA), with `--base google/madlad400-7b-mt --batch-size 4 --grad-accum 8`.
2. If it runs out of memory even with `--batch-size 1 --grad-accum 32`, drop E6 (no quantisation) and record a `Ruling:`.
3. If it fits, launch `e6` with the same scheme as `run_e5.sh` (without `--full`, with `--lr 2e-4`) and bring back the results.

- [ ] **Step 2: Commit the E6 results** (if any)

```bash
git add results/
git commit -m "docs: resultados E6 (LoRA de MADLAD-7B)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Conclusions**

In `results/SUMMARY.md`, before `## Conclusions (subproject 1b, …)`, insert a section `## Conclusions (subproject 1c, <date>)` with:
- A table of NLLB-1.3B base (reference, CC-BY-NC) / MADLAD-3B base / MADLAD-7B base / E5 / E6 × FLORES es→gn (chrF++, BLEU), FLORES gn→es (chrF++, BLEU), `jojajovai-anotado` gn→es chrF++, with a licence column and ✅/❌ per threshold.
- Which sources were excluded by licence and how many examples remained (from `models/e5/train_meta.json`).
- The effect of full fine-tuning on MADLAD-3B (E5 − MADLAD-3B base) and a comparison with the best NLLB (E3). Say whether chrF++ and BLEU agree.
- The duration of each phase.
- The best **commercially usable** model, its distance to the thresholds, and the next step.

- [ ] **Step 4: Commit**

```bash
git add results/
git commit -m "docs: conclusiones del subproyecto 1c

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
