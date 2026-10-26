# Subproject 2 — Commercial STT (Omnilingual and Whisper) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a commercially licensed Guarani speech recogniser that matches or beats MMS (CER ≤ 5.8 %, WER ≤ 41.6 % on Common Voice 17 gn test), comparing Omnilingual ASR (with and without fine-tuning) and fine-tuned Whisper large-v3-turbo.

**Architecture:**
- `stt.transcribe` picks a backend from the model name. MMS and Whisper run in the main environment (`transformers`); Omnilingual runs in a separate uv environment (`envs/omnilingual/`, fairseq2) called via `subprocess`.
- Common Voice data is loaded and decontaminated in `train/stt_data.py`.
- Whisper is fine-tuned with `Seq2SeqTrainer` (`train/whisper.py`). Omnilingual uses the official fairseq2 recipe, fed by a parquet that `train/omni.py` exports.

**Tech Stack:** Python 3.12, torch 2.14 + CUDA 13, transformers 4.57, librosa, soundfile, pyarrow, jiwer; a separate environment with `omnilingual-asr` 0.2 (fairseq2 ≤ 0.6).

**Spec:** `docs/superpowers/specs/2026-09-27-subproject-2-stt-commercial-design.md`

## Global Constraints

- Commercial use: only the base models `openai/whisper-*` (MIT) and `facebook/omniASR-*` / `omniASR_*` cards (Apache 2.0); data only from Common Voice (CC0).
- Minimum threshold: CER ≤ 5.8 % and WER ≤ 41.6 % on `cv` test (855 clips); target WER ≤ 35 %.
- `cv` test never enters training or dev: every sentence whose `decont_key` is in test is dropped.
- The best checkpoint is picked by CER on `cv` dev, including the step-0 evaluation.
- The main environment does not install fairseq2; Omnilingual lives only in `envs/omnilingual/`.
- Data or configuration errors in the CLI → exit code 2 (`DataMissing` / `ValueError`).
- Fast tests: `uv run pytest -q`; slow: `uv run pytest -q -m slow`.
- Training runs on the remote server `user@gpu-server` (`~/Guanani`, RTX PRO 5000 24 GB):
  - sync the code with `rsync -a --exclude .venv --exclude models --exclude data --exclude results --exclude .superpowers --exclude 'envs/omnilingual/.venv' --exclude 'envs/omnilingual/repo' ./ user@gpu-server:~/Guanani/`;
  - launch with `setsid nohup … > log 2>&1 < /dev/null & disown`;
  - bring back only the new rows of `results/SUMMARY.md` (never overwrite the local SUMMARY with rsync).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not touch the user's processes on the local 3060 (llama-server, server.py).

## Review Focus

1. An unreadable or empty Common Voice clip in train/other (`other` has ~19k clips) must not bring down a training run that takes hours: it is dropped and counted. → tests in Task 5 (`usable_clips`) and Task 4 (`export`).
2. A fine-tuned model in `models/<name>/` must transcribe with the right backend given nothing but its path (Whisper via `config.json`, Omnilingual via `card.yaml`). → tests in Task 2 and Task 3.
3. Long audio without silences (music, continuous noise) must not produce a chunk > 30 s: Whisper would truncate it and Omnilingual rejects it (> 40 s). → test in Task 2 (`split_long` with continuous noise).
4. If the Omnilingual environment is missing, the user must get the exact command to create it, not a `FileNotFoundError` from subprocess. → test in Task 3.
5. CER: lower is better. Whisper's dev callback must not copy the chrF logic (higher is better) from `lora.DevChrF`. → test in Task 5 (`DevCER` picks the minimum).

---

### Task 1: Common Voice data for STT

**Files:**
- Modify: `scripts/download.sh` (Common Voice block)
- Modify: `src/guanani/data.py` (dataset `cv-dev`)
- Modify: `src/guanani/eval.py:69` (`COMPATIBLE["stt"]`)
- Create: `src/guanani/train/stt_data.py`
- Modify: `docs/data-licenses.md`
- Test: `tests/test_stt_data.py`

**Interfaces:**
- Consumes: `guanani.data.load_common_voice(split) -> list[{"audio": str, "gn": str}]`, `guanani.train.corpus.decont_key(text) -> str`, `guanani.normalize.normalize(text) -> str`.
- Produces: `stt_data.SPLITS = ("train", "dev", "other")`; `stt_data.load_stt(kinds: list[str]) -> tuple[list[dict], dict]` with rows `{"audio": str, "text": str, "split": str}` and statistics `{split: {"in", "vacia", "contaminada", "aceptada"}}`; dataset `"cv-dev"` in `data.DATASETS`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stt_data.py
import pytest

from guanani.train import stt_data

HEAD = "client_id\tpath\tsentence\n"


def _cv(tmp_path, monkeypatch, **splits):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    d = tmp_path / "common_voice"
    d.mkdir()
    for split, sentences in splits.items():
        rows = "".join(f"c\t{split}_{i}.mp3\t{s}\n" for i, s in enumerate(sentences))
        (d / f"{split}.tsv").write_text(HEAD + rows, encoding="utf-8")


def test_load_stt_drops_sentences_seen_in_test(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, test=["Mba'éichapa reiko?"], train=["mba’éichapa reiko", "Che areko peteĩ jagua."])
    rows, stats = stt_data.load_stt(["train"])
    assert [r["text"] for r in rows] == ["Che areko peteĩ jagua."]
    assert stats["train"] == {"in": 2, "vacia": 0, "contaminada": 1, "aceptada": 1}


def test_load_stt_normalizes_text_and_keeps_split(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, test=["x"], dev=["Ñande  rekove  âĝuã"])
    rows, _ = stt_data.load_stt(["dev"])
    assert rows[0]["text"] == "Ñande rekove ãg̃uã"
    assert rows[0]["split"] == "dev"
    assert rows[0]["audio"].endswith("clips/dev_0.mp3")


def test_load_stt_counts_empty_sentences(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, test=["x"], other=["   "])
    rows, stats = stt_data.load_stt(["other"])
    assert rows == [] and stats["other"]["vacia"] == 1


def test_load_stt_rejects_test_and_unknown_splits(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, test=["x"])
    with pytest.raises(ValueError, match="test"):
        stt_data.load_stt(["test"])
    with pytest.raises(ValueError, match="foo"):
        stt_data.load_stt(["foo"])


def test_cv_dev_dataset_is_registered(tmp_path, monkeypatch):
    from guanani import data, eval as ev

    _cv(tmp_path, monkeypatch, dev=["a"])
    assert data.DATASETS["cv-dev"]()[0]["gn"] == "a"
    assert "cv-dev" in ev.COMPATIBLE["stt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stt_data.py -q`
Expected: FAIL (`ImportError: cannot import name 'stt_data'`).

- [ ] **Step 3: Implement**

```python
# src/guanani/train/stt_data.py
"""Common Voice para entrenar STT: frases normalizadas y descontaminadas frente al test."""

from guanani import data
from guanani.normalize import normalize
from guanani.train.corpus import decont_key

SPLITS = ("train", "dev", "other")


def load_stt(kinds: list[str]) -> tuple[list[dict], dict]:
    for k in kinds:
        if k not in SPLITS:
            raise ValueError(f"split de Common Voice no válido para entrenar: {k}; opciones: {list(SPLITS)}")
    test_keys = {decont_key(r["gn"]) for r in data.load_common_voice("test")}
    rows, stats = [], {}
    for k in kinds:
        s = dict.fromkeys(("in", "vacia", "contaminada", "aceptada"), 0)
        for r in data.load_common_voice(k):
            s["in"] += 1
            text = normalize(r["gn"])
            if not text:
                s["vacia"] += 1
            elif decont_key(text) in test_keys:
                s["contaminada"] += 1
            else:
                s["aceptada"] += 1
                rows.append({"audio": r["audio"], "text": text, "split": k})
        stats[k] = s
    return rows, stats
```

In `src/guanani/data.py`, add to `DATASETS` (after `"cv"`):

```python
    "cv-dev": lambda: load_common_voice("dev"),
```

In `src/guanani/eval.py`, change `"stt": {"cv"},` to:

```python
    "stt": {"cv", "cv-dev"},
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stt_data.py -q`
Expected: 5 passed. If `test_load_stt_normalizes_text_and_keeps_split` fails on the nasal mapping, check `normalize._TABLE` maps `â→ã`, `ĝ→g̃` (it does) — do not change `normalize`.

- [ ] **Step 5: Download train/dev/other**

Replace the Common Voice heredoc body in `scripts/download.sh` (the part between `repo = ...` and `print("Common Voice OK")`) with:

```python
repo = "fsicoli/common_voice_17_0"
d = Path("data/common_voice"); (d / "clips").mkdir(parents=True, exist_ok=True)
from huggingface_hub import HfApi
files = HfApi().list_repo_files(repo, repo_type="dataset")
for split in ("test", "dev", "train", "other"):
    shutil.copy(hf_hub_download(repo, f"transcript/gn/{split}.tsv", repo_type="dataset"), d / f"{split}.tsv")
    for tar in sorted(f for f in files if f.startswith(f"audio/gn/{split}/") and f.endswith(".tar")):
        with tarfile.open(hf_hub_download(repo, tar, repo_type="dataset")) as t:
            for m in t.getmembers():
                if m.isfile():
                    m.name = Path(m.name).name
                    t.extract(m, d / "clips", filter="data")
```

Run: `bash scripts/download.sh 2>&1 | tail -5` then `wc -l data/common_voice/*.tsv && ls data/common_voice/clips | wc -l`
Expected: `train.tsv` 1572, `dev.tsv` 361, `test.tsv` 856, `other.tsv` 18779 lines (with header); ~21,500 clips.

- [ ] **Step 6: Licenses**

In `docs/data-licenses.md`, first table, add a row after `jojajovai`:

```markdown
| `cv17` | Common Voice 17 gn (réplica `fsicoli/common_voice_17_0`) | CC0 | sí | Entrenamiento STT (train, dev, other). Voces de donantes; no intentar identificar hablantes. | https://commonvoice.mozilla.org |
```

In "Evaluation only", change the Common Voice row to `| Common Voice 17 (gn), test split | CC0 | https://commonvoice.mozilla.org |`. In "Models and tools", add:

```markdown
| Omnilingual ASR (CTC y LLM, v2) | Apache 2.0 | candidatos comerciales STT |
| Whisper large-v3-turbo | MIT | candidato comercial STT |
```

- [ ] **Step 7: Run the whole fast suite and commit**

Run: `uv run pytest -q`
Expected: all pass (99 passed).

```bash
git add scripts/download.sh src/guanani/data.py src/guanani/eval.py src/guanani/train/stt_data.py tests/test_stt_data.py docs/data-licenses.md
git commit -m "feat: datos de Common Voice para STT, descontaminados frente al test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: STT backends (Whisper) and long audio

**Files:**
- Modify: `src/guanani/stt.py`
- Modify: `src/guanani/cli.py` (`transcribe --long`)
- Test: `tests/test_stt.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `stt.backend_of(model: str) -> Literal["mms", "whisper", "omnilingual"]`; `stt.split_long(audio: np.ndarray, max_s: float = 30.0, top_db: int = 35) -> list[np.ndarray]`; `stt.transcribe_long(audio, model=DEFAULT_MODEL, device=None) -> str`; `stt.transcribe(audios, model, device, batch_size=16)` dispatches by backend. `WHISPER_LANG = "es"`. Task 3 adds the `omnilingual` branch of `transcribe`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_stt.py
import json

import numpy as np

from guanani import stt


def test_backend_of_hub_names():
    assert stt.backend_of("facebook/mms-1b-all") == "mms"
    assert stt.backend_of("openai/whisper-large-v3-turbo") == "whisper"
    assert stt.backend_of("omniASR_CTC_1B_v2") == "omnilingual"


def test_backend_of_local_dirs(tmp_path):
    w = tmp_path / "w1"
    w.mkdir()
    (w / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    o = tmp_path / "s1"
    o.mkdir()
    (o / "card.yaml").write_text("name: guanani_s1\n")
    m = tmp_path / "m"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"model_type": "wav2vec2"}))
    assert [stt.backend_of(str(p)) for p in (w, o, m)] == ["whisper", "omnilingual", "mms"]


def _tone(seconds):
    t = np.arange(int(seconds * stt.SR)) / stt.SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_split_long_cuts_on_silence():
    gap = np.zeros(stt.SR, dtype=np.float32)
    audio = np.concatenate([_tone(20), gap, _tone(20), gap, _tone(5)])
    chunks = stt.split_long(audio)
    assert len(chunks) == 2
    assert all(len(c) <= 30 * stt.SR for c in chunks)
    assert sum(len(c) for c in chunks) >= 44 * stt.SR


def test_split_long_hard_cuts_audio_without_silence():
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, 75 * stt.SR).astype(np.float32)
    chunks = stt.split_long(noise)
    assert len(chunks) == 3 and max(len(c) for c in chunks) <= 30 * stt.SR


def test_transcribe_long_joins_chunk_texts(monkeypatch):
    seen = []
    monkeypatch.setattr(stt, "transcribe", lambda audios, model, device: seen.append(len(audios)) or ["a", " ", "b"])
    gap = np.zeros(stt.SR, dtype=np.float32)
    assert stt.transcribe_long(np.concatenate([_tone(25), gap, _tone(25), gap, _tone(25)])) == "a b"
    assert seen == [3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stt.py -q`
Expected: FAIL (`AttributeError: module 'guanani.stt' has no attribute 'backend_of'`).

- [ ] **Step 3: Implement**

Replace `src/guanani/stt.py` with:

```python
"""Reconocimiento de voz en guaraní: MMS (referencia no comercial), Whisper y Omnilingual."""

import json
from functools import cache
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

from guanani.device import pick_device

SR = 16000
DEFAULT_MODEL = "facebook/mms-1b-all"
WHISPER_LANG = "es"  # Whisper no tiene guaraní: se afina reutilizando la etiqueta del español.
MAX_CHUNK_S = 30.0


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if audio.size == 0:
        raise ValueError(f"audio vacío: {path}")
    return audio


def backend_of(model: str) -> str:
    p = Path(model)
    if (p / "card.yaml").exists() or model.startswith("omniASR_"):  # tarjetas fairseq2, p. ej. omniASR_CTC_1B_v2
        return "omnilingual"
    if (p / "config.json").exists():
        kind = json.loads((p / "config.json").read_text(encoding="utf-8")).get("model_type")
        return "whisper" if kind == "whisper" else "mms"
    return "whisper" if "whisper" in model else "mms"


@cache
def _load_mms(model: str, device: str):
    proc = AutoProcessor.from_pretrained(model, target_lang="grn")
    m = Wav2Vec2ForCTC.from_pretrained(model, target_lang="grn", ignore_mismatched_sizes=True).to(device).eval()
    return proc, m


def _mms(audios, model, device):
    proc, m = _load_mms(model, device)
    out = []
    # ponytail: de a uno, sin padding; MMS es solo referencia.
    for a in audios:
        inputs = proc(a, sampling_rate=SR, return_tensors="pt").to(device)
        with torch.inference_mode():
            ids = m(**inputs).logits.argmax(-1)[0]
        out.append(proc.decode(ids))
    return out


@cache
def _load_whisper(model: str, device: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    m = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=dtype).to(device).eval()
    return proc, m


def _whisper(audios, model, device, batch_size):
    proc, m = _load_whisper(model, device)
    out, i = [], 0
    while i < len(audios):
        batch = audios[i:i + batch_size]
        feats = proc(batch, sampling_rate=SR, return_tensors="pt").input_features.to(device, m.dtype)
        try:
            with torch.inference_mode():
                ids = m.generate(feats, language=WHISPER_LANG, task="transcribe", num_beams=1, max_new_tokens=225)
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            ids = None
        if ids is None:  # fuera del except: así se libera la memoria del intento fallido
            batch_size //= 2
            torch.cuda.empty_cache()
            continue
        out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
        i += len(batch)
    return out


def transcribe(audios: list[np.ndarray], model: str = DEFAULT_MODEL, device: str | None = None,
               batch_size: int = 16) -> list[str]:
    device = pick_device(device)
    backend = backend_of(model)
    if backend == "whisper":
        return _whisper(audios, model, device, batch_size)
    if backend == "omnilingual":
        return _omnilingual(audios, model, batch_size)
    return _mms(audios, model, device)


def split_long(audio: np.ndarray, max_s: float = MAX_CHUNK_S, top_db: int = 35) -> list[np.ndarray]:
    max_n = int(max_s * SR)
    chunks, cur, cur_n = [], [], 0
    for s, e in librosa.effects.split(audio, top_db=top_db):
        for i in range(s, e, max_n):  # tramos sin silencio de más de max_s: corte duro
            piece = audio[i:min(i + max_n, e)]
            if cur and cur_n + len(piece) > max_n:
                chunks.append(np.concatenate(cur))
                cur, cur_n = [], 0
            cur.append(piece)
            cur_n += len(piece)
    if cur:
        chunks.append(np.concatenate(cur))
    return chunks


def transcribe_long(audio: np.ndarray, model: str = DEFAULT_MODEL, device: str | None = None) -> str:
    # ponytail: VAD por energía (librosa); un VAD neuronal si el habla con ruido de fondo se corta mal.
    texts = transcribe(split_long(audio), model, device)
    return " ".join(t.strip() for t in texts if t.strip())
```

Temporary stub so the module imports until Task 3 (place above `transcribe`):

```python
def _omnilingual(audios, model, batch_size):
    raise NotImplementedError("backend omnilingual: Task 3")
```

In `src/guanani/cli.py`, add to the `transcribe` parser `s.add_argument("--long", action="store_true", help="long audio: split on silences")` and replace its branch body with:

```python
            from guanani.stt import load_audio, transcribe, transcribe_long

            audio = load_audio(a.audio)
            print(transcribe_long(audio, device=a.device, **_kw(a)) if a.long
                  else transcribe([audio], device=a.device, **_kw(a))[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stt.py -q`
Expected: 5 passed. In `test_split_long_cuts_on_silence` the expected grouping is [20 s + 1 gap-trimmed 20 s → exceeds 30 → split] so chunks = [20 s], [20 s + 5 s]; if librosa keeps a few ms of silence the length assertion (`>= 44 s`) still holds.

- [ ] **Step 5: Check the MMS path still works and commit**

Run: `uv run pytest -q && uv run pytest -q -m slow -k "stt or roundtrip"`
Expected: fast suite all pass; slow STT/TTS roundtrip passes (MMS unchanged).

```bash
git add src/guanani/stt.py src/guanani/cli.py tests/test_stt.py
git commit -m "feat: backend Whisper en stt y transcripción de audios largos por silencios

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Omnilingual environment and subprocess backend

**Files:**
- Create: `envs/omnilingual/pyproject.toml`
- Create: `envs/omnilingual/transcribe.py`
- Modify: `src/guanani/stt.py` (`_omnilingual`)
- Modify: `.gitignore`
- Test: `tests/test_stt.py`, `tests/test_models_slow.py`

**Interfaces:**
- Consumes: `stt.backend_of`, `stt.SR`.
- Produces: `stt.OMNI_ENV: Path` (override with env var `GUANANI_OMNI_ENV`); `stt.OMNI_LANG = "grn_Latn"`; `stt._omnilingual(audios, model, batch_size) -> list[str]`. Script protocol: `<env>/.venv/bin/python <env>/transcribe.py --model CARD [--asset-dir DIR] [--lang L] --batch-size N IN.npz OUT.jsonl`, one JSON line `{"text": ...}` per audio, in order. A local model is a directory with a `card.yaml` whose card is named `guanani_<directory name>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stt.py`:

```python
import os
import sys

import pytest


def _fake_env(tmp_path):
    env = tmp_path / "omni"
    (env / ".venv" / "bin").mkdir(parents=True)
    py = env / ".venv" / "bin" / "python"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    (env / "transcribe.py").write_text(
        "import argparse, json, numpy as np\n"
        "p = argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--asset-dir')\n"
        "p.add_argument('--lang'); p.add_argument('--batch-size'); p.add_argument('inp'); p.add_argument('out')\n"
        "a = p.parse_args(); n = len(np.load(a.inp).files)\n"
        "open(a.out, 'w').write(''.join(json.dumps({'text': f'{a.model}|{a.asset_dir}|{a.lang}|{i}'}) + '\\n' for i in range(n)))\n"
    )
    return env


def test_omnilingual_runs_script_in_its_env(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "OMNI_ENV", _fake_env(tmp_path))
    out = stt.transcribe([np.zeros(10, np.float32)] * 2, model="omniASR_LLM_1B_v2")
    assert out == ["omniASR_LLM_1B_v2|None|grn_Latn|0", "omniASR_LLM_1B_v2|None|grn_Latn|1"]


def test_omnilingual_ctc_gets_no_lang_and_local_card(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "OMNI_ENV", _fake_env(tmp_path))
    model = tmp_path / "s1"
    model.mkdir()
    (model / "card.yaml").write_text("name: guanani_s1\n")
    assert stt.transcribe([np.zeros(10, np.float32)], model=str(model)) == [f"guanani_s1|{model}|None|0"]


def test_omnilingual_missing_env_explains_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "OMNI_ENV", tmp_path / "nope")
    with pytest.raises(RuntimeError, match="uv sync --project envs/omnilingual"):
        stt.transcribe([np.zeros(10, np.float32)], model="omniASR_CTC_300M_v2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stt.py -q -k omnilingual`
Expected: FAIL (`AttributeError: ... 'OMNI_ENV'`).

- [ ] **Step 3: Implement the backend**

In `src/guanani/stt.py`, add imports `import os`, `import subprocess`, `import tempfile`; constants:

```python
OMNI_ENV = Path(os.environ.get("GUANANI_OMNI_ENV", Path(__file__).resolve().parents[2] / "envs" / "omnilingual"))
OMNI_LANG = "grn_Latn"  # código con el que Omnilingual entrenó guaraní (38,6 h según su tabla por idioma)
```

Replace the `_omnilingual` stub with:

```python
def _omnilingual(audios, model, batch_size):
    # fairseq2 exige otro torch: Omnilingual vive en su propio entorno y se llama por subprocess.
    python = OMNI_ENV / ".venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"Falta el entorno de Omnilingual en {OMNI_ENV}: ejecuta `uv sync --project envs/omnilingual`")
    p = Path(model)
    card, asset_dir = (f"guanani_{p.name}", str(p)) if (p / "card.yaml").exists() else (model, None)
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.npz", Path(tmp) / "out.jsonl"
        np.savez(inp, *audios)
        cmd = [str(python), str(OMNI_ENV / "transcribe.py"), "--model", card, "--batch-size", str(batch_size)]
        if asset_dir:
            cmd += ["--asset-dir", asset_dir]
        if "_LLM_" in card:
            cmd += ["--lang", OMNI_LANG]
        subprocess.run(cmd + [str(inp), str(out)], check=True)
        return [json.loads(line)["text"] for line in out.read_text(encoding="utf-8").splitlines()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stt.py -q`
Expected: 8 passed.

- [ ] **Step 5: Create the separate environment**

`envs/omnilingual/pyproject.toml`:

```toml
[project]
name = "guanani-omnilingual"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = ["omnilingual-asr==0.2.0", "numpy", "soundfile"]

[tool.uv]
package = false
```

`envs/omnilingual/transcribe.py`:

```python
"""Transcribe con Omnilingual ASR. Se ejecuta en envs/omnilingual/.venv (fairseq2), no en el entorno principal."""

import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="tarjeta fairseq2, p. ej. omniASR_CTC_1B_v2 o guanani_s1")
p.add_argument("--asset-dir", help="directorio con card.yaml de un modelo afinado")
p.add_argument("--lang", help="código de idioma para modelos LLM, p. ej. grn_Latn")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("inp")
p.add_argument("out")
a = p.parse_args()
if a.asset_dir:
    os.environ["FAIRSEQ2_USER_ASSET_DIR"] = a.asset_dir  # antes de importar fairseq2

import torch  # noqa: E402
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: E402

audios = [{"waveform": torch.from_numpy(x), "sample_rate": 16000} for x in np.load(a.inp).values()]
pipe = ASRInferencePipeline(model_card=a.model)
kw = {"lang": [a.lang] * len(audios)} if a.lang else {}
texts = pipe.transcribe(audios, batch_size=a.batch_size, **kw)
with open(a.out, "w", encoding="utf-8") as f:
    f.writelines(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts)
```

Append to `.gitignore`:

```
envs/omnilingual/.venv/
envs/omnilingual/repo/
```

Run on the remote server (after rsync, see Global Constraints): `ssh user@gpu-server 'cd ~/Guanani && uv sync --project envs/omnilingual 2>&1 | tail -3 && envs/omnilingual/.venv/bin/python -c "import torch, fairseq2; print(torch.__version__, torch.cuda.is_available(), fairseq2.__version__)"'`
Expected: prints torch version, `True`, fairseq2 0.5.x/0.6.x. If `cuda.is_available()` is False (fairseq2's torch wheel lacks CUDA 13 kernels for Blackwell), add to the env's pyproject a `[tool.uv.sources] torch = { index = "pytorch-cu128" }` with `[[tool.uv.index]] name = "pytorch-cu128" url = "https://download.pytorch.org/whl/cu128" explicit = true`, pin the torch version fairseq2 requires, re-sync, and ledger a Ruling.

- [ ] **Step 6: Slow test (real model) and FAIRSEQ2_USER_ASSET_DIR check**

Append to `tests/test_models_slow.py`:

```python
@pytest.mark.slow
def test_omnilingual_ctc_transcribes_and_local_card_works(tmp_path):
    from guanani import stt

    if not (stt.OMNI_ENV / ".venv" / "bin" / "python").exists():
        pytest.skip("sin entorno envs/omnilingual")
    audio = stt.load_audio(str(sorted((stt.Path("data/common_voice/clips")).glob("*.mp3"))[0]))
    base = stt.transcribe([audio], model="omniASR_CTC_300M_v2")[0]
    card = tmp_path / "alias"
    card.mkdir()
    (card / "card.yaml").write_text(
        "name: guanani_alias\nmodel_family: wav2vec2_asr\nmodel_arch: 300m_v2\n"
        "checkpoint: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt\n"
        "tokenizer_ref: omniASR_tokenizer_written_v2\n"
    )
    assert base.strip() and stt.transcribe([audio], model=str(card))[0] == base
```

Run on the server: `ssh user@gpu-server 'cd ~/Guanani && uv run pytest -q -m slow -k omnilingual'`
Expected: 1 passed. If the local card is not found, fairseq2 uses a different variable for user asset dirs: check `envs/omnilingual/.venv/lib/python*/site-packages/fairseq2` with `grep -rn "ASSET_DIR" --include=*.py | head`, use the variable it reads, and ledger a Ruling.

- [ ] **Step 7: Commit**

Run: `uv run pytest -q` (all pass).

```bash
git add envs/omnilingual/pyproject.toml envs/omnilingual/transcribe.py envs/omnilingual/uv.lock src/guanani/stt.py .gitignore tests/test_stt.py tests/test_models_slow.py
git commit -m "feat: backend Omnilingual ASR en entorno aparte (fairseq2) llamado por subprocess

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(`envs/omnilingual/uv.lock` comes from the server: `scp user@gpu-server:~/Guanani/envs/omnilingual/uv.lock envs/omnilingual/`.)

---

### Task 4: Omnilingual fine-tuning (export, recipe, card, best checkpoint)

**Files:**
- Create: `src/guanani/train/omni.py`
- Create: `envs/omnilingual/ctc-finetune.yaml`
- Create: `scripts/omni_finetune.sh`
- Modify: `src/guanani/cli.py` (`train omni-export`, `train omni-pick`)
- Test: `tests/test_omni.py`

**Interfaces:**
- Consumes: `stt_data.load_stt(kinds)`, `stt.load_audio`, `stt.transcribe`, `eval.score_asr`, `data.load_common_voice("dev")`.
- Produces: `omni.export(rows: list[dict], split: str, root: Path) -> dict` (`{"filas", "descartadas", "horas"}`), writes `root/version=0/corpus=cv17/split=<split>/language=grn_Latn/part-0.parquet`; `omni.write_stats(root, hours) -> Path` (`root/language_distribution_0.tsv`); `omni.write_card(model_dir: Path, checkpoint: str, arch: str) -> Path`; `omni.pick_best(name: str, run_dir: Path, arch: str, models_dir=Path("models"), dev_limit=None) -> dict` (evaluates each `checkpoints/step_*` on cv dev by CER and leaves the best one's card in `models/<name>/card.yaml`; step 0 = base model).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_omni.py
import io

import numpy as np
import pyarrow.dataset as pads
import soundfile as sf

from guanani.train import omni


def _wav(path, seconds):
    sf.write(path, np.full(int(seconds * 16000), 0.1, np.float32), 16000)
    return str(path)


def test_export_writes_hive_parquet_with_flac_bytes(tmp_path):
    rows = [{"audio": _wav(tmp_path / "a.wav", 1.0), "text": "che"}, {"audio": str(tmp_path / "missing.wav"), "text": "x"}]
    stats = omni.export(rows, "train", tmp_path / "ds")
    assert stats == {"filas": 1, "descartadas": 1, "horas": round(1 / 3600, 6)}
    t = pads.dataset(tmp_path / "ds" / "version=0", partitioning="hive").to_table()
    r = t.to_pylist()[0]
    assert (r["text"], r["audio_size"], r["corpus"], r["split"], r["language"]) == ("che", 16000, "cv17", "train", "grn_Latn")
    audio, sr = sf.read(io.BytesIO(np.array(r["audio_bytes"], np.int8).tobytes()))
    assert sr == 16000 and len(audio) == 16000


def test_write_stats_tsv(tmp_path):
    p = omni.write_stats(tmp_path, 2.5)
    assert p.read_text() == "corpus\tlanguage\thours\ncv17\tgrn_Latn\t2.5\n"


def test_write_card_names_model_after_dir(tmp_path):
    p = omni.write_card(tmp_path / "s1", "/runs/s1/checkpoints/step_500", "1b_v2")
    text = p.read_text()
    assert "name: guanani_s1\n" in text and "model_arch: 1b_v2\n" in text
    assert "checkpoint: /runs/s1/checkpoints/step_500\n" in text


def test_pick_best_prefers_lowest_cer_including_base(tmp_path, monkeypatch):
    run = tmp_path / "run"
    for s in (500, 1000):
        (run / "checkpoints" / f"step_{s}").mkdir(parents=True)
    cers = {"omniASR_CTC_1B_v2": 0.08, "500": 0.05, "1000": 0.06}
    monkeypatch.setattr(omni, "dev_cer", lambda model, rows: cers[model if model.startswith("omniASR") else model.rsplit("_step_", 1)[1]])
    monkeypatch.setattr(omni, "dev_rows", lambda limit: [])
    out = omni.pick_best("s1", run, "1b_v2", models_dir=tmp_path / "models")
    assert out["best_step"] == 500 and out["history"]["0"] == 0.08
    assert "step_500" in (tmp_path / "models" / "s1" / "card.yaml").read_text()
```

`pick_best` evaluates each checkpoint through a temporary card dir named `<name>_step_<N>` (see Step 3); the fake reads the step from that name.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_omni.py -q`
Expected: FAIL (`ImportError: cannot import name 'omni'`).

- [ ] **Step 3: Implement**

```python
# src/guanani/train/omni.py
"""Afinado de Omnilingual ASR: exporta Common Voice al parquet de su receta y elige el mejor punto por CER en dev."""

import io
import json
import shutil
from pathlib import Path

import numpy as np

CORPUS, LANG, SR = "cv17", "grn_Latn", 16000
BASES = {"300m_v2": "omniASR_CTC_300M_v2", "1b_v2": "omniASR_CTC_1B_v2", "3b_v2": "omniASR_CTC_3B_v2"}
BASE_URLS = {
    "300m_v2": "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt",
    "1b_v2": "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B-v2.pt",
    "3b_v2": "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-3B-v2.pt",
}


def export(rows: list[dict], split: str, root: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    from guanani.stt import load_audio

    texts, audio, sizes, dropped = [], [], [], 0
    for r in rows:
        try:
            a = load_audio(r["audio"])
        except Exception:  # clip ilegible o vacío: se descarta, no se aborta la exportación
            dropped += 1
            continue
        buf = io.BytesIO()
        sf.write(buf, a, SR, format="FLAC")
        texts.append(r["text"])
        audio.append(np.frombuffer(buf.getvalue(), dtype=np.int8))
        sizes.append(len(a))
    d = root / "version=0" / f"corpus={CORPUS}" / f"split={split}" / f"language={LANG}"
    d.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "text": pa.array(texts, pa.string()),
        "audio_bytes": pa.array(audio, pa.list_(pa.int8())),
        "audio_size": pa.array(sizes, pa.int64()),
    })
    pq.write_table(table, d / "part-0.parquet", row_group_size=100)
    return {"filas": len(texts), "descartadas": dropped, "horas": round(sum(sizes) / SR / 3600, 6)}


def write_stats(root: Path, hours: float) -> Path:
    p = root / "language_distribution_0.tsv"
    p.write_text(f"corpus\tlanguage\thours\n{CORPUS}\t{LANG}\t{hours}\n", encoding="utf-8")
    return p


def write_card(model_dir: Path, checkpoint: str, arch: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    p = model_dir / "card.yaml"
    p.write_text(
        f"name: guanani_{model_dir.name}\nmodel_family: wav2vec2_asr\nmodel_arch: {arch}\n"
        f"checkpoint: {checkpoint}\ntokenizer_ref: omniASR_tokenizer_written_v2\n",
        encoding="utf-8",
    )
    return p


def dev_rows(limit: int | None) -> list[dict]:
    from guanani import data

    return data.load_common_voice("dev")[:limit]


def dev_cer(model: str, rows: list[dict]) -> float:
    from guanani.eval import load_clips, score_asr
    from guanani.stt import transcribe

    audios, refs, _ = load_clips(rows)
    return score_asr(transcribe(audios, model=model), refs)["cer"]


def pick_best(name: str, run_dir: Path, arch: str, models_dir: Path = Path("models"), dev_limit: int | None = None) -> dict:
    rows = dev_rows(dev_limit)
    history = {"0": dev_cer(BASES[arch], rows)}
    steps = sorted((int(p.name.removeprefix("step_")), p) for p in (run_dir / "checkpoints").glob("step_*"))
    tmp = models_dir / f".{name}_eval"
    for step, path in steps:
        card_dir = tmp / f"{name}_step_{step}"
        write_card(card_dir, str(path.resolve()), arch)
        history[str(step)] = dev_cer(str(card_dir), rows)
        print(f"[dev] paso {step}: CER {history[str(step)]}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    best_step = int(min(history, key=history.get))
    out = models_dir / name
    # Si gana la base (paso 0), la tarjeta apunta al checkpoint publicado.
    ckpt = BASE_URLS[arch] if best_step == 0 else str((run_dir / "checkpoints" / f"step_{best_step}").resolve())
    write_card(out, ckpt, arch)
    meta = {"base": BASES[arch], "run_dir": str(run_dir), "best_step": best_step, "best_cer": history[str(best_step)], "history": history}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
```

CLI (`src/guanani/cli.py`, inside the `train` subparsers):

```python
    oe = tsub.add_parser("omni-export", help="exporta Common Voice al parquet de la receta de Omnilingual")
    oe.add_argument("--name", required=True)
    oe.add_argument("--data", default="train", help="train o train,other")
    op = tsub.add_parser("omni-pick", help="elige el mejor checkpoint de Omnilingual por CER en cv dev")
    op.add_argument("--name", required=True)
    op.add_argument("--run-dir", required=True)
    op.add_argument("--arch", choices=["300m_v2", "1b_v2", "3b_v2"], required=True)
    op.add_argument("--dev-limit", type=int)
```

and branches:

```python
            elif a.train_cmd == "omni-export":
                from pathlib import Path

                from guanani.train import omni, stt_data

                root = Path("data/train/omni") / a.name
                train_rows, stats = stt_data.load_stt(a.data.split(","))
                dev_rows, _ = stt_data.load_stt(["dev"])
                out = {"train": omni.export(train_rows, "train", root), "dev": omni.export(dev_rows, "dev", root),
                       "filtrado": stats}
                omni.write_stats(root, out["train"]["horas"])
                print(json.dumps(out, ensure_ascii=False, indent=2))
            elif a.train_cmd == "omni-pick":
                from pathlib import Path

                from guanani.train.omni import pick_best

                print(json.dumps(pick_best(a.name, Path(a.run_dir), a.arch, dev_limit=a.dev_limit), indent=2))
```

Note: `omni-export` with `--data train,other` exports dev decontaminated as well (dev is not in `--data`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_omni.py -q`
Expected: 4 passed.

- [ ] **Step 5: Recipe config and launcher**

`envs/omnilingual/ctc-finetune.yaml` (placeholders `@MODEL@`, `@DATA@` filled by the script):

```yaml
model:
  name: "@MODEL@"

dataset:
  name: "guanani_cv17"
  train_split: "train"
  valid_split: "dev"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "@DATA@/language_distribution_0.tsv"
    beta_corpus: 0.5
    beta_language: 0.5
    fragment_loading:
      cache: True
  asr_task_config:
    min_audio_len: 16_000
    max_audio_len: 960_000
    max_num_elements: 7_680_000
    batch_shuffle_window: 1
    normalize_audio: true
    example_shuffle_window: 0

tokenizer:
  name: "omniASR_tokenizer_written_v2"

optimizer:
  config:
    lr: 1e-05

trainer:
  freeze_encoder_for_n_steps: 0
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 4

regime:
  num_steps: 5_000
  validate_every_n_steps: 500
  validate_after_n_steps: 500
  checkpoint_every_n_steps: 500
  checkpoint_after_n_steps: 500
  keep_last_n_checkpoints: 20
  publish_metrics_every_n_steps: 100
```

`scripts/omni_finetune.sh`:

```bash
#!/usr/bin/env bash
# Uso: scripts/omni_finetune.sh <name> <model_card> [data_name]
# Afina Omnilingual con la receta oficial de fairseq2 y deja el run en runs/<name>.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; card=$2; data=$(realpath "data/train/omni/${3:-$name}")
env=envs/omnilingual
[ -d $env/repo ] || git clone --depth 1 https://github.com/facebookresearch/omnilingual-asr $env/repo
assets=$(realpath runs/$name/assets); mkdir -p "$assets"
cat > "$assets/guanani_cv17.yaml" <<EOF
name: guanani_cv17
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: $data/version=0
tokenizer_ref: omniASR_tokenizer_written_v2
EOF
sed -e "s|@MODEL@|$card|" -e "s|@DATA@|$data|" $env/ctc-finetune.yaml > "runs/$name/config.yaml"
cd $env/repo
FAIRSEQ2_USER_ASSET_DIR="$assets" ../.venv/bin/python -m workflows.recipes.wav2vec2.asr \
  "$(realpath ../../../runs/$name)" --config-file "$(realpath ../../../runs/$name/config.yaml)"
```

Add `runs/` to `.gitignore`.

- [ ] **Step 6: Smoke run on the server**

On the server (after rsync): `uv run guanani train omni-export --name smoke --data train` then edit nothing and run `sed -i 's/num_steps: 5_000/num_steps: 20/; s/_every_n_steps: 500/_every_n_steps: 10/; s/_after_n_steps: 500/_after_n_steps: 10/' envs/omnilingual/ctc-finetune.yaml` **on the server copy only**, then `bash scripts/omni_finetune.sh smoke omniASR_CTC_300M_v2 2>&1 | tail -20; ls runs/smoke/checkpoints`.
Expected: 20 steps run, `runs/smoke/checkpoints/step_10` and `step_20` exist. Then `uv run guanani train omni-pick --name smoke --run-dir runs/smoke --arch 300m_v2 --dev-limit 20`.
Expected: JSON with `history` for steps 0, 10, 20. If the checkpoint layout under `step_N` is a directory tree that `load_model` cannot read from the card's `checkpoint:` path, find the model file (`find runs/smoke/checkpoints/step_10 -name "*.pt"`) and point `pick_best`'s card at it (adjust the path in `pick_best` and its test), ledger a Ruling. If the dataset card key differs (error mentions `dataset_config`), check `envs/omnilingual/repo/src/omnilingual_asr/datasets/impl/mixture_parquet_asr_dataset.py` for the expected config field and adjust the heredoc; ledger a Ruling. Restore the yaml afterwards (`git checkout envs/omnilingual/ctc-finetune.yaml` on the server, or re-rsync) and `rm -rf runs/smoke models/smoke data/train/omni/smoke`.

- [ ] **Step 7: Commit**

Run: `uv run pytest -q` (all pass).

```bash
git add src/guanani/train/omni.py src/guanani/cli.py tests/test_omni.py envs/omnilingual/ctc-finetune.yaml scripts/omni_finetune.sh .gitignore
git commit -m "feat: afinado de Omnilingual (exportación parquet, receta fairseq2, mejor punto por CER)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Whisper fine-tuning

**Files:**
- Create: `src/guanani/train/whisper.py`
- Modify: `src/guanani/cli.py` (`train stt`)
- Test: `tests/test_whisper.py`, `tests/test_models_slow.py`

**Interfaces:**
- Consumes: `stt_data.load_stt(kinds)`, `stt.load_audio`, `stt.SR`, `stt.WHISPER_LANG`, `eval.score_asr`, `lora.prepare_out_dir(out, resume)`.
- Produces: `whisper.BASE = "openai/whisper-large-v3-turbo"`; `whisper.check_base(base)`; `whisper.usable_clips(rows, max_s=30.0) -> tuple[list[dict], dict]`; `whisper.Collator(proc)`; `whisper.DevCER(proc, dev_rows, eval_steps, out_dir)`; `whisper.run(name, data_kinds=("train",), base=BASE, epochs=20.0, batch_size=16, grad_accum=2, lr=1e-5, max_steps=-1, eval_steps=100, resume=False, models_dir=Path("models"), dev_limit=None) -> dict`. The best model ends up in `models/<name>/best_model/` (with the processor) and `models/<name>/train_meta.json`.

Ruling already taken in this plan (vs. spec "bf16, Adafactor"): Whisper-turbo (809M) fits in 24 GB with fp32 master weights + bf16 autocast + AdamW; pure bf16 lost small updates in subproject 1c (E5), so we avoid it. Cost if wrong: slower steps, nothing else.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_whisper.py
import json

import numpy as np
import pytest
import soundfile as sf

from guanani.train import whisper


def test_check_base_rejects_non_whisper():
    whisper.check_base("openai/whisper-large-v3-turbo")
    with pytest.raises(ValueError, match="openai/whisper-"):
        whisper.check_base("facebook/mms-1b-all")


def test_usable_clips_drops_long_empty_and_broken(tmp_path):
    def wav(name, seconds):
        sf.write(tmp_path / name, np.full(int(seconds * 16000), 0.1, np.float32), 16000)
        return str(tmp_path / name)

    rows = [{"audio": wav("ok.wav", 2), "text": "a"}, {"audio": wav("long.wav", 31), "text": "b"},
            {"audio": wav("empty.wav", 0), "text": "c"}, {"audio": str(tmp_path / "nope.mp3"), "text": "d"}]
    kept, stats = whisper.usable_clips(rows)
    assert [r["text"] for r in kept] == ["a"]
    assert stats == {"in": 4, "largo": 1, "ilegible": 2, "aceptado": 1}


def test_collator_masks_padding_and_strips_start_token():
    from transformers import WhisperProcessor

    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="spanish", task="transcribe")
    col = whisper.Collator(proc)
    feats = [{"input_features": np.zeros((80, 3000), np.float32), "labels": proc.tokenizer(t).input_ids}
             for t in ("che", "che areko peteĩ jagua")]
    batch = col(feats)
    sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    assert (batch["labels"][:, 0] != sot).all()
    assert (batch["labels"][0] == -100).any() and not (batch["labels"][1] == -100).any()


def test_dev_cer_keeps_lowest(tmp_path):
    (tmp_path / "dev_history.json").write_text(json.dumps({"0": 0.09, "100": 0.05, "200": 0.07}))
    cb = whisper.DevCER(proc=None, dev_rows=[], eval_steps=100, out_dir=tmp_path)
    assert (cb.best, cb.best_step) == (0.05, 100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_whisper.py -q`
Expected: FAIL (`ImportError: cannot import name 'whisper'`).

- [ ] **Step 3: Implement**

```python
# src/guanani/train/whisper.py
"""Afinado completo de Whisper para guaraní (etiqueta de idioma es), mejor punto por CER en Common Voice dev."""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from guanani.stt import SR, WHISPER_LANG, load_audio

BASE = "openai/whisper-large-v3-turbo"
LANG_NAME = "spanish"  # nombre que espera WhisperProcessor para WHISPER_LANG


def check_base(base: str) -> None:
    if not base.startswith("openai/whisper-"):
        raise ValueError(f"modelo base no permitido para uso comercial: {base} (solo openai/whisper-*)")


def usable_clips(rows: list[dict], max_s: float = 30.0) -> tuple[list[dict], dict]:
    # ponytail: decodifica cada clip una vez para validarlo; con other (~19k clips) tarda unos minutos.
    stats = dict.fromkeys(("in", "largo", "ilegible", "aceptado"), 0)
    kept = []
    for r in rows:
        stats["in"] += 1
        try:
            n = len(load_audio(r["audio"]))
        except Exception:
            stats["ilegible"] += 1
            continue
        if n > max_s * SR:
            stats["largo"] += 1
        else:
            stats["aceptado"] += 1
            kept.append(r)
    return kept, stats


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, rows, proc):
        self.rows, self.proc = rows, proc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feats = self.proc.feature_extractor(load_audio(r["audio"]), sampling_rate=SR).input_features[0]
        return {"input_features": feats, "labels": self.proc.tokenizer(r["text"]).input_ids}


class Collator:
    def __init__(self, proc):
        self.proc = proc
        self.sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, feats):
        batch = self.proc.feature_extractor.pad([{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = self.proc.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.sot).all():  # el modelo antepone el token de inicio por su cuenta
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class DevCER(TrainerCallback):
    """Evalúa CER en dev cada eval_steps (y en el paso 0) y guarda el mejor modelo. Menor es mejor."""

    def __init__(self, proc, dev_rows: list[dict], eval_steps: int, out_dir: Path):
        self.proc, self.dev, self.eval_steps, self.out_dir = proc, dev_rows, eval_steps, out_dir
        self.path = out_dir / "dev_history.json"
        self.history = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.best, self.best_step = float("inf"), None
        for step, cer in self.history.items():
            if cer < self.best:
                self.best, self.best_step = cer, int(step)

    def _cer(self, model) -> float:
        from guanani.eval import score_asr

        was_training = model.training
        model.eval()
        hyps = []
        for i in range(0, len(self.dev), 16):
            audios = [load_audio(r["audio"]) for r in self.dev[i:i + 16]]
            feats = self.proc(audios, sampling_rate=SR, return_tensors="pt").input_features.to(model.device, model.dtype)
            with torch.inference_mode(), torch.autocast(model.device.type, dtype=torch.bfloat16, enabled=model.device.type == "cuda"):
                ids = model.generate(feats, language=WHISPER_LANG, task="transcribe", num_beams=1, max_new_tokens=225)
            hyps += self.proc.batch_decode(ids, skip_special_tokens=True)
        model.train(was_training)
        return score_asr(hyps, [r["text"] for r in self.dev])["cer"]

    def _evaluate(self, model, step: int) -> None:
        cer = self._cer(model)
        self.history[str(step)] = cer
        if cer < self.best:
            self.best, self.best_step = cer, step
            model.save_pretrained(self.out_dir / "best_model")
            self.proc.save_pretrained(self.out_dir / "best_model")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        print(f"[dev] paso {step}: CER {cer} (mejor {self.best} en {self.best_step})", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kw):
        if "0" not in self.history:
            self._evaluate(model, 0)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.eval_steps == 0:
            self._evaluate(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kw):
        if str(state.global_step) not in self.history:
            self._evaluate(model, state.global_step)


def run(name: str, data_kinds=("train",), base: str = BASE, epochs: float = 20.0, batch_size: int = 16,
        grad_accum: int = 2, lr: float = 1e-5, max_steps: int = -1, eval_steps: int = 100, resume: bool = False,
        models_dir: Path = Path("models"), dev_limit: int | None = None) -> dict:
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

    from guanani.train import stt_data
    from guanani.train.lora import prepare_out_dir

    check_base(base)
    rows, filt = stt_data.load_stt(list(data_kinds))
    dev, _ = stt_data.load_stt(["dev"])
    train_rows, clip_stats = usable_clips(rows)
    dev_rows, _ = usable_clips(dev[:dev_limit])
    out = models_dir / name
    out.mkdir(parents=True, exist_ok=True)
    prepare_out_dir(out, resume)

    proc = WhisperProcessor.from_pretrained(base, language=LANG_NAME, task="transcribe")
    # fp32 + autocast bf16: en bf16 puro se perdían las actualizaciones pequeñas (subproyecto 1c).
    model = WhisperForConditionalGeneration.from_pretrained(base)
    model.config.apply_spec_augment, model.config.mask_time_prob = True, 0.05
    model.generation_config.language, model.generation_config.task = WHISPER_LANG, "transcribe"
    model.generation_config.forced_decoder_ids = None

    cuda = torch.cuda.is_available()
    cb = DevCER(proc, dev_rows, eval_steps, out)
    args = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=50,
        num_train_epochs=epochs,
        max_steps=max_steps,
        bf16=cuda,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4 if cuda else 0,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ClipDataset(train_rows, proc),
                             data_collator=Collator(proc), callbacks=[cb])
    trainer.train(resume_from_checkpoint=True if resume else None)
    meta = {"base": base, "data": list(data_kinds), "filtrado": filt, "clips": clip_stats, "dev": len(dev_rows),
            "best_step": cb.best_step, "best_cer": cb.best, "history": cb.history}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
```

CLI (`src/guanani/cli.py`, inside `train` subparsers):

```python
    st = tsub.add_parser("stt", help="afina Whisper (models/<name>/best_model)")
    st.add_argument("--name", required=True)
    st.add_argument("--data", default="train", help="train o train,other")
    st.add_argument("--base", default="openai/whisper-large-v3-turbo")
    st.add_argument("--epochs", type=float, default=20.0)
    st.add_argument("--batch-size", type=int, default=16)
    st.add_argument("--grad-accum", type=int, default=2)
    st.add_argument("--lr", type=float, default=1e-5)
    st.add_argument("--max-steps", type=int, default=-1)
    st.add_argument("--eval-steps", type=int, default=100)
    st.add_argument("--dev-limit", type=int)
    st.add_argument("--resume", action="store_true")
```

Branch (before the final `else` of `train`, which is the NMT one — change that `else` to `elif a.train_cmd in ("nmt", "lora"):`):

```python
            elif a.train_cmd == "stt":
                from guanani.train.whisper import run as run_whisper

                meta = run_whisper(a.name, a.data.split(","), base=a.base, epochs=a.epochs, batch_size=a.batch_size,
                                   grad_accum=a.grad_accum, lr=a.lr, max_steps=a.max_steps, eval_steps=a.eval_steps,
                                   resume=a.resume, dev_limit=a.dev_limit)
                print(json.dumps({k: meta[k] for k in ("best_step", "best_cer")}))
```

Add to `tests/test_cli.py`:

```python
def test_train_stt_rejects_non_whisper_base(capsys):
    from guanani.cli import main

    assert main(["train", "stt", "--name", "x", "--base", "facebook/mms-1b-all"]) == 2
    assert "openai/whisper-" in capsys.readouterr().err
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_whisper.py tests/test_cli.py -q`
Expected: all pass.

- [ ] **Step 5: Slow test (tiny Whisper, a few steps)**

Append to `tests/test_models_slow.py`:

```python
@pytest.mark.slow
def test_whisper_tiny_finetune_saves_best_and_transcribes(tmp_path, monkeypatch):
    from guanani import stt
    from guanani.train import stt_data, whisper

    rows, _ = stt_data.load_stt(["dev"])
    monkeypatch.setattr(stt_data, "load_stt", lambda kinds: (rows[:8], {}))
    meta = whisper.run("tiny", base="openai/whisper-tiny", max_steps=4, eval_steps=2, batch_size=4, grad_accum=1,
                       lr=1e-4, models_dir=tmp_path, dev_limit=4)
    assert set(meta["history"]) == {"0", "2", "4"}
    best = tmp_path / "tiny" / "best_model"
    assert stt.backend_of(str(best)) == "whisper"
    assert isinstance(stt.transcribe([stt.load_audio(rows[0]["audio"])], model=str(best))[0], str)
```

Run: `uv run pytest -q -m slow -k whisper_tiny` (local 3060, ~1 GB VRAM; the user's services keep running).
Expected: 1 passed.

- [ ] **Step 6: Commit**

Run: `uv run pytest -q` (all pass).

```bash
git add src/guanani/train/whisper.py src/guanani/cli.py tests/test_whisper.py tests/test_cli.py tests/test_models_slow.py
git commit -m "feat: afinado de Whisper con mejor punto por CER en Common Voice dev

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Experiments S0–S2, W1–W2 and conclusions

**Files:**
- Create: `scripts/long_audio.py`
- Modify: `src/guanani/stt.py` (`DEFAULT_MODEL` → best commercial model)
- Modify: `results/SUMMARY.md`
- Modify: memory file `guanani-project-decisions.md`

**Interfaces:**
- Consumes: everything above; CLI `guanani eval stt cv|cv-dev --model …`, `guanani train stt|omni-export|omni-pick`, `scripts/omni_finetune.sh`.
- Produces: rows in `results/SUMMARY.md`, models `models/s1`, `models/s2`, `models/w1`, `models/w2` on the server, conclusions section.

- [ ] **Step 1: Sync and download data on the server**

```bash
rsync -a --exclude .venv --exclude models --exclude data --exclude results --exclude .superpowers --exclude 'envs/omnilingual/.venv' --exclude 'envs/omnilingual/repo' ./ user@gpu-server:~/Guanani/
ssh user@gpu-server 'cd ~/Guanani && uv sync && bash scripts/download.sh 2>&1 | grep -i "common voice"; wc -l data/common_voice/*.tsv'
```

Expected: `Common Voice OK`; line counts as in Task 1 Step 5.

- [ ] **Step 2: S0 — Omnilingual without fine-tuning**

```bash
ssh user@gpu-server 'cd ~/Guanani && setsid nohup bash -c "for m in omniASR_CTC_300M_v2 omniASR_CTC_1B_v2 omniASR_CTC_3B_v2 omniASR_LLM_1B_v2; do uv run guanani eval stt cv --model \$m; done" > s0.log 2>&1 < /dev/null & disown'
```

Expected when done (`tail s0.log`): four result paths; each row in the server's `results/SUMMARY.md` with wer/cer. Pick the S1 size: the lowest-CER CTC model whose inference fits ≤ 8 GB (300M ~2 GB, 1B ~3 GB, 3B ~8 GB per the Omnilingual table). If LLM-1B is clearly best (> 1 CER point better than every CTC), note it in the conclusions (its fine-tuning recipe is `llm-finetune.yaml`, out of scope for this plan).

- [ ] **Step 3: S1 — fine-tune the chosen CTC on CV train**

```bash
ssh user@gpu-server 'cd ~/Guanani && uv run guanani train omni-export --name s1 --data train && setsid nohup bash -c "bash scripts/omni_finetune.sh s1 <CARD> && uv run guanani train omni-pick --name s1 --run-dir runs/s1 --arch <ARCH> && uv run guanani eval stt cv --model models/s1" > s1.log 2>&1 < /dev/null & disown'
```

(`<CARD>`/`<ARCH>` from Step 2, e.g. `omniASR_CTC_1B_v2` / `1b_v2`.)
Expected: `models/s1/train_meta.json` with `best_step` and dev history; a `cv` row for `models/s1`.

- [ ] **Step 4: W1 — Whisper large-v3-turbo on CV train (in parallel with S1 only if VRAM allows; otherwise after)**

```bash
ssh user@gpu-server 'cd ~/Guanani && setsid nohup bash -c "uv run guanani train stt --name w1 --data train && uv run guanani eval stt cv --model models/w1/best_model" > w1.log 2>&1 < /dev/null & disown'
```

Expected: ~1,000 steps (1,571 clips / 32 × 20 epochs); dev CER printed every 100 steps; `cv` row for `models/w1/best_model`. If step 0 dev CER is > 0.9 and the curve is still falling steeply at the end, rerun with `--epochs 40 --name w1b` and record both.

- [ ] **Step 5: S2 / W2 — add `other`**

```bash
ssh user@gpu-server 'cd ~/Guanani && uv run guanani train omni-export --name s2 --data train,other && setsid nohup bash -c "bash scripts/omni_finetune.sh s2 <CARD> && uv run guanani train omni-pick --name s2 --run-dir runs/s2 --arch <ARCH> && uv run guanani eval stt cv --model models/s2" > s2.log 2>&1 < /dev/null & disown'
```

Run W2 only if S2's test CER < S1's: `uv run guanani train stt --name w2 --data train,other --epochs 3` then eval `models/w2/best_model`.

- [ ] **Step 6: Long-audio control**

```python
# scripts/long_audio.py
"""Control de audios largos: concatena clips de cv test en audios de ~90 s y mide WER con transcribe_long."""

import sys

import numpy as np

from guanani import data
from guanani.eval import load_clips, score_asr
from guanani.stt import SR, transcribe_long

model, n_audios = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
audios, refs, _ = load_clips(data.load_common_voice("test"))
gap = np.zeros(SR // 2, np.float32)
hyps, joined, i = [], [], 0
for _ in range(n_audios):
    parts, texts = [], []
    while i < len(audios) and sum(len(p) for p in parts) < 90 * SR:
        parts += [audios[i], gap]
        texts.append(refs[i])
        i += 1
    hyps.append(transcribe_long(np.concatenate(parts), model=model))
    joined.append(" ".join(texts))
print(model, score_asr(hyps, joined))
```

Run on the server for the best commercial model: `uv run python scripts/long_audio.py <best model>` and for MMS as reference.
Expected: WER close to the clip-level WER (± a few points); a big gap means `split_long` cuts words.

- [ ] **Step 7: Results and conclusions**

Bring back only the new rows: `ssh user@gpu-server 'cd ~/Guanani && grep "| stt |" results/SUMMARY.md' ` and append to the local table the rows not already present. Add a section "Subproject 2 — commercial STT" above the subproject 1d conclusions in `results/SUMMARY.md` with: table S0–S2/W1–W2 (model, license, test CER/WER, dev best step), threshold check (CER ≤ 5.8 %, WER ≤ 41.6 %, target WER ≤ 35 %), long-audio WER, contamination caveat (Omnilingual and MMS may have seen Common Voice), and the recommended model. Set `DEFAULT_MODEL` in `src/guanani/stt.py` to the recommended commercial model **only if it meets the minimum threshold**; otherwise keep MMS and say so.

- [ ] **Step 8: Commit and update memory**

Run: `uv run pytest -q` (all pass).

```bash
git add scripts/long_audio.py src/guanani/stt.py results/SUMMARY.md
git commit -m "docs: resultados y conclusiones del subproyecto 2 (STT comercial)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Update `~/.claude/projects/-home-josemi-Documents-Personal-Ideas-Guanani/memory/guanani-project-decisions.md` with a "2 (STT)" line: best commercial model, CER/WER, where it lives.
