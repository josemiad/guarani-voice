# Subproject 3 — Commercial TTS (Piper from scratch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a commercially licensed Guarani voice from scratch (Piper/VITS) from the Common Voice speaker with the most material, and compare it with MMS-TTS on intelligibility (round-trip CER with W2), naturalness (UTMOS) and CPU speed.

**Architecture:**
- `train/tts_data.py` prepares the voice: W2 filter, 22,050 Hz audio, Piper's `metadata.csv`.
- Piper trains in a separate uv environment (`envs/piper`, a pinned clone of `piper1-gpl`) via `scripts/piper_train.sh`.
- `train/piper.py` exports the checkpoints to ONNX and picks the best by round-trip CER on FLORES dev.
- `tts.speak` synthesises with `piper-tts` (onnxruntime) in the main environment when the model is a directory with a `.onnx`.

**Tech Stack:** Python 3.12, `piper-tts` 1.8.0 (inference), `piper1-gpl` commit `efffbfb226bfb511ebbcf55d0cecd8b35a89743d` (training, Lightning 2), librosa, soundfile, SpeechMOS `utmos22_strong` via `torch.hub`.

**Spec:** `docs/superpowers/specs/2026-09-30-subproject-3-tts-commercial-design.md`

## Global Constraints

- Training from scratch only: never `--ckpt_path` from published Piper voices (they inherit "lessac", a research licence) nor MMS-TTS weights (CC-BY-NC).
- Data: Common Voice 17 gn (CC0) only. Voice = the speaker with the most clips, unless `--speaker` is given.
- Thresholds (first 200 FLORES+ devtest sentences): round-trip CER with W2 ≤ T0's; UTMOS ≥ T0's; CPU RTF ≤ 0.2.
- Checkpoint selection by round-trip CER (W2) on 100 FLORES+ dev sentences; lower is better.
- FLORES+ dev and devtest sentences never enter training.
- Piper (GPL-3.0) is used only on the server for training and export; the main environment installs only `piper-tts`, for inference.
- Data or configuration errors in the CLI → exit code 2.
- Fast tests: `uv run pytest -q`; slow: `uv run pytest -q -m slow`.
- Server: `user@gpu-server` (`~/Guanani`).
  - Check `nvidia-smi` before launching. If another user's process is on the GPU, ask the user to stop it; never kill it.
  - Sync with `rsync -a --exclude .venv --exclude models --exclude data --exclude results --exclude runs --exclude .superpowers --exclude 'envs/*/.venv' --exclude 'envs/*/repo' ./ user@gpu-server:~/Guanani/`.
  - Launch with `setsid nohup … > log 2>&1 < /dev/null & disown`.
  - Don't use `pkill -f` inside an `ssh` call: it kills the session itself.
  - Bring back only the new rows of `results/SUMMARY.md`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A clip that is unreadable, empty or that W2 cannot transcribe must not abort preparation (hours with 5,500 clips): it is dropped and counted. → test in Task 1 (`prepare_voice` with a broken clip).
2. Synthesis text with digits, odd punctuation or only whitespace: `speak` must synthesise what it can or raise a clear `ValueError`, never an onnxruntime error. → test in Task 2 (`speak` with empty text on the piper backend).
3. A voice directory without `.onnx.json` (only `.onnx`) must not silently be treated as MMS. → test in Task 2 (`backend_of`).
4. `tts-pick` on a run without checkpoints must fail with a clear message, not pick "nothing". → test in Task 4.
5. CER: lower is better when picking a checkpoint (do not copy the UTMOS/`val_mos` logic, where higher is better). → test in Task 4.

---

### Task 1: Voice preparation

**Files:**
- Create: `src/guanani/train/tts_data.py`
- Modify: `src/guanani/cli.py` (`train tts-prep`)
- Modify: `docs/data-licenses.md`
- Test: `tests/test_tts_data.py`

**Interfaces:**
- Consumes: `data.load_common_voice(split) -> [{"audio", "gn", "speaker"}]`, `normalize.normalize`, `corpus.decont_key`, `data.load_flores(split)`, `stt.load_audio(path) -> np.ndarray` (16 kHz), `stt.transcribe(audios, model)`, `eval.score_asr`.
- Produces: `tts_data.SR = 22050`; `tts_data.top_speaker() -> str`; `tts_data.text_reason(text: str, eval_keys: set[str]) -> str | None`; `tts_data.audio_reason(audio: np.ndarray, sr: int) -> str | None`; `tts_data.to_voice_audio(path: str) -> np.ndarray` (22,050 Hz, −20 dBFS); `tts_data.prepare_voice(name: str, speaker: str | None = None, asr_model: str = "models/w2", max_cer: float = 0.10, out_root: Path = Path("data/train/tts"), transcribe=None) -> dict`. Output: `data/train/tts/<name>/{wavs/*.wav, metadata.csv, stats.json}`; `metadata.csv` = `file.wav|text` with no header.

Ruling vs. spec (already taken here): Piper's data module already does the train/validation split (`validation_split`) and silence trimming (Silero VAD), so we do **not** write `val.csv` nor trim ourselves; we pass `--data.validation_split 0.03`. Trimming is only used to measure duration for the 1–15 s filter. Cost if wrong: validation may share a sentence with training (a single speaker rarely repeats sentences).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tts_data.py
import csv
import json

import numpy as np
import pytest
import soundfile as sf

from guanani.train import tts_data

HEAD = "client_id\tpath\tsentence\n"


def _cv(tmp_path, monkeypatch, rows_by_split):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    d = tmp_path / "common_voice"
    (d / "clips").mkdir(parents=True)
    for split, rows in rows_by_split.items():
        (d / f"{split}.tsv").write_text(HEAD + "".join(f"{s}\t{p}\t{t}\n" for s, p, t in rows), encoding="utf-8")
    f = tmp_path / "flores"
    f.mkdir()
    for split in ("dev", "devtest"):
        for code in ("gug_Latn", "spa_Latn"):
            (f / f"{split}.{code}.jsonl").write_text(json.dumps({"id": 1, "text": f"Frase de {split}"}), encoding="utf-8")
    return d / "clips"


def _wav(path, seconds, amp=0.1, sr=48000):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)


def test_text_reason():
    keys = {"frase de dev"}
    assert tts_data.text_reason("Che róga", keys) is None
    assert tts_data.text_reason("   ", keys) == "vacia"
    assert tts_data.text_reason("Ary 2024-pe", keys) == "cifras"
    assert tts_data.text_reason("Frase de dev.", keys) == "flores"


def test_audio_reason():
    sr = tts_data.SR
    tone = lambda s, a=0.1: (a * np.sin(np.arange(int(s * sr)) / 7)).astype(np.float32)
    assert tts_data.audio_reason(tone(3), sr) is None
    assert tts_data.audio_reason(tone(0.5), sr) == "corto"
    assert tts_data.audio_reason(tone(16), sr) == "largo"
    clipped = tone(3, a=1.5).clip(-1, 1)
    assert tts_data.audio_reason(clipped, sr) == "saturado"


def test_to_voice_audio_resamples_and_normalizes(tmp_path):
    _wav(tmp_path / "a.wav", 2.0, amp=0.5)
    a = tts_data.to_voice_audio(str(tmp_path / "a.wav"))
    rms_db = 20 * np.log10(np.sqrt(np.mean(a**2)))
    assert abs(len(a) - 2 * tts_data.SR) < 50
    assert abs(rms_db - (-20)) < 0.5


def test_top_speaker_counts_all_splits(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, {"test": [("a", "1.mp3", "x")], "dev": [("b", "2.mp3", "y")],
                                "train": [("b", "3.mp3", "z")], "other": [("c", "4.mp3", "w")]})
    assert tts_data.top_speaker() == "b"


def test_prepare_voice_filters_and_writes_piper_metadata(tmp_path, monkeypatch):
    clips = _cv(tmp_path, monkeypatch, {
        "test": [], "dev": [], "train": [
            ("v", "ok.wav", "Che róga porã"),
            ("v", "misread.wav", "Mba'éichapa"),
            ("v", "broken.wav", "Ndaipóri"),
            ("v", "digits.wav", "Ary 1990"),
            ("w", "other_speaker.wav", "Ambue"),
        ], "other": []})
    for name in ("ok", "misread", "digits", "other_speaker"):
        _wav(clips / f"{name}.wav", 3.0)
    (clips / "broken.wav").write_bytes(b"not audio")
    fake = {"Che róga porã": "che róga porã", "Mba'éichapa": "otra cosa totalmente distinta"}
    out = tts_data.prepare_voice("v1", speaker="v", out_root=tmp_path / "tts",
                                 transcribe=lambda texts_audios: [fake[t] for t, _ in texts_audios])
    rows = list(csv.reader((tmp_path / "tts" / "v1" / "metadata.csv").open(encoding="utf-8"), delimiter="|"))
    assert rows == [["ok.wav", "Che róga porã"]]
    assert (tmp_path / "tts" / "v1" / "wavs" / "ok.wav").exists()
    assert sf.info(tmp_path / "tts" / "v1" / "wavs" / "ok.wav").samplerate == tts_data.SR
    assert out["descartes"] == {"cifras": 1, "ilegible": 1, "asr": 1}
    assert out["aceptados"] == 1 and out["horas"] > 0
    assert json.loads((tmp_path / "tts" / "v1" / "stats.json").read_text())["aceptados"] == 1


def test_prepare_voice_unknown_speaker_raises(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, {"test": [], "dev": [], "train": [("v", "a.wav", "x")], "other": []})
    with pytest.raises(ValueError, match="nadie"):
        tts_data.prepare_voice("v1", speaker="nadie", out_root=tmp_path / "tts", transcribe=lambda x: [])
```

`transcribe` is injected for tests: it receives `[(text, audio16k), ...]` for the clips that passed the text and audio filters and returns one hypothesis per item. The default (None) uses `stt.transcribe` with `asr_model`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tts_data.py -q`
Expected: FAIL (`ImportError: cannot import name 'tts_data'`).

- [ ] **Step 3: Implement**

```python
# src/guanani/train/tts_data.py
"""Voz para Piper: clips de un hablante de Common Voice, filtrados por W2, a 22.050 Hz y −20 dBFS."""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from guanani import data
from guanani.normalize import normalize
from guanani.train.corpus import decont_key

SR = 22050
SPLITS = ("train", "dev", "test", "other")
TARGET_DBFS = -20.0


def top_speaker() -> str:
    counts = Counter(r["speaker"] for s in SPLITS for r in data.load_common_voice(s))
    return counts.most_common(1)[0][0]


def text_reason(text: str, eval_keys: set[str]) -> str | None:
    if not text.strip():
        return "vacia"
    if re.search(r"\d", text):
        return "cifras"  # no hay lectura fiable de cifras en guaraní
    if decont_key(text) in eval_keys:
        return "flores"
    return None


def audio_reason(audio: np.ndarray, sr: int) -> str | None:
    import librosa

    if np.mean(np.abs(audio) >= 0.999) >= 0.001:
        return "saturado"
    seconds = len(librosa.effects.trim(audio, top_db=30)[0]) / sr
    if seconds < 1.0:
        return "corto"
    if seconds > 15.0:
        return "largo"
    return None


def to_voice_audio(path: str) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=SR, mono=True)
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        raise ValueError(f"audio en silencio: {path}")
    gain = 10 ** (TARGET_DBFS / 20) / rms
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def _default_transcribe(asr_model: str):
    from guanani.stt import SR as ASR_SR
    from guanani.stt import transcribe

    def run(items):
        import librosa

        return transcribe([librosa.resample(a, orig_sr=SR, target_sr=ASR_SR) for _, a in items], model=asr_model)

    return run


def prepare_voice(name: str, speaker: str | None = None, asr_model: str = "models/w2", max_cer: float = 0.10,
                  out_root: Path = Path("data/train/tts"), transcribe=None) -> dict:
    import soundfile as sf

    from guanani.eval import score_asr

    speaker = speaker or top_speaker()
    rows = [r for s in SPLITS for r in data.load_common_voice(s) if r["speaker"] == speaker]
    if not rows:
        raise ValueError(f"el hablante {speaker} no tiene clips en Common Voice")
    eval_keys = {decont_key(r["gn"]) for s in ("dev", "devtest") for r in data.load_flores(s)}
    transcribe = transcribe or _default_transcribe(asr_model)

    drops, seen, candidates = Counter(), set(), []
    for r in rows:
        text = normalize(r["gn"])
        reason = text_reason(text, eval_keys) or ("repetida" if decont_key(text) in seen else None)
        if reason is None:
            try:
                audio = to_voice_audio(r["audio"])
                reason = audio_reason(audio, SR)
            except Exception:  # clip ilegible o en silencio: se descarta, no se aborta
                reason = "ilegible"
        if reason:
            drops[reason] += 1
            continue
        seen.add(decont_key(text))
        candidates.append((Path(r["audio"]).stem + ".wav", text, audio))

    # ponytail: una sola llamada al ASR con todos los clips; con ~5.500 clips cabe en memoria (float32 22 kHz ≈ 2 GB).
    hyps = transcribe([(t, a) for _, t, a in candidates]) if candidates else []
    out = out_root / name
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    kept, seconds = [], 0.0
    for (fname, text, audio), hyp in zip(candidates, hyps):
        if score_asr([hyp], [text])["cer"] > max_cer:
            drops["asr"] += 1
            continue
        sf.write(out / "wavs" / fname, audio, SR)
        kept.append(f"{fname}|{text}\n")
        seconds += len(audio) / SR
    (out / "metadata.csv").write_text("".join(kept), encoding="utf-8")
    stats = {"hablante": speaker, "entrada": len(rows), "aceptados": len(kept), "horas": round(seconds / 3600, 3),
             "descartes": dict(drops), "asr_model": asr_model, "max_cer": max_cer}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats
```

CLI (`src/guanani/cli.py`, inside `train` subparsers, before `a = p.parse_args(argv)`):

```python
    tp = tsub.add_parser("tts-prep", help="prepara la voz de un hablante de Common Voice para Piper")
    tp.add_argument("--name", required=True)
    tp.add_argument("--speaker", help="client_id; por defecto el hablante con más clips")
    tp.add_argument("--asr-model", default="models/w2")
    tp.add_argument("--max-cer", type=float, default=0.10)
```

Branch (before `elif a.train_cmd in ("nmt", "lora"):`):

```python
            elif a.train_cmd == "tts-prep":
                from guanani.train.tts_data import prepare_voice

                print(json.dumps(prepare_voice(a.name, a.speaker, a.asr_model, a.max_cer), ensure_ascii=False, indent=2))
```

Note in the test `test_prepare_voice_filters_and_writes_piper_metadata`: `misread` fails the ASR filter; `digits` is dropped by text; `broken` is unreadable; the other speaker's clip is never considered.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tts_data.py -q`
Expected: 6 passed.

- [ ] **Step 5: Licenses**

In `docs/data-licenses.md`, extend the `cv17` row's "Conditions and risks" with: `Also the TTS prototype voice (the speaker with the most clips): a donor who did not expressly consent to a synthesiser; replace with a professional voice actor before launch.` In "Models and tools", add:

```markdown
| Piper (`piper1-gpl`, entrenamiento) | GPL-3.0 | solo en el servidor para entrenar y exportar; no se distribuye; las voces entrenadas no son GPL |
| `piper-tts` (inferencia ONNX) | GPL-3.0 | uso en servidor (sin distribución) |
| UTMOS (`tarepan/SpeechMOS`, `utmos22_strong`) | MIT (código) | solo evaluación |
```

- [ ] **Step 6: Commit**

Run: `uv run pytest -q` (all pass).

```bash
git add src/guanani/train/tts_data.py src/guanani/cli.py tests/test_tts_data.py docs/data-licenses.md
git commit -m "feat: preparación de la voz TTS desde Common Voice (filtro W2, 22 kHz, metadata de Piper)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Piper backend in `tts.speak`

**Files:**
- Modify: `pyproject.toml` (dependency `piper-tts==1.8.0`)
- Modify: `src/guanani/tts.py`
- Test: `tests/test_tts.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `tts.backend_of(model: str) -> Literal["piper", "mms"]` (raises `ValueError` if a directory has `.onnx` without `.onnx.json`); `tts.speak(text, model=DEFAULT_MODEL, device=None) -> tuple[np.ndarray, int]` dispatches by backend; `tts.voice_file(model: str) -> Path` (the directory's `.onnx`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tts.py
import pytest

from guanani import tts


def test_backend_of(tmp_path):
    v = tmp_path / "t1"
    v.mkdir()
    (v / "t1.onnx").write_bytes(b"x")
    (v / "t1.onnx.json").write_text("{}")
    assert tts.backend_of(str(v)) == "piper"
    assert tts.voice_file(str(v)) == v / "t1.onnx"
    assert tts.backend_of("facebook/mms-tts-grn") == "mms"


def test_backend_of_onnx_without_config_is_an_error(tmp_path):
    v = tmp_path / "t1"
    v.mkdir()
    (v / "t1.onnx").write_bytes(b"x")
    with pytest.raises(ValueError, match="onnx.json"):
        tts.backend_of(str(v))


def test_speak_empty_text_raises_before_loading_voice(tmp_path):
    v = tmp_path / "t1"
    v.mkdir()
    (v / "t1.onnx").write_bytes(b"not a model")
    (v / "t1.onnx.json").write_text("{}")
    with pytest.raises(ValueError, match="vacío"):
        tts.speak("  ", model=str(v))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tts.py -q`
Expected: FAIL (`AttributeError: module 'guanani.tts' has no attribute 'backend_of'`).

- [ ] **Step 3: Implement**

Add to `pyproject.toml` dependencies: `"piper-tts==1.8.0",` then `uv sync`.

Replace `src/guanani/tts.py` with:

```python
"""Síntesis de voz en guaraní: Piper (ONNX, voces propias) y MMS-TTS (referencia no comercial)."""

from functools import cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from guanani.device import pick_device
from guanani.normalize import normalize

DEFAULT_MODEL = "facebook/mms-tts-grn"


def voice_file(model: str) -> Path:
    return next(Path(model).glob("*.onnx"))


def backend_of(model: str) -> str:
    p = Path(model)
    if p.is_dir() and any(p.glob("*.onnx")):
        onnx = voice_file(model)
        if not onnx.with_suffix(".onnx.json").exists():
            raise ValueError(f"falta {onnx.name}.json junto a {onnx}: exporta la voz con `guanani train tts-pick`")
        return "piper"
    return "mms"


@cache
def _load_mms(model: str, device: str):
    return AutoTokenizer.from_pretrained(model), VitsModel.from_pretrained(model).to(device).eval()


@cache
def _load_piper(model: str, use_cuda: bool):
    from piper import PiperVoice

    return PiperVoice.load(str(voice_file(model)), use_cuda=use_cuda)


def _mms(text, model, device):
    tok, m = _load_mms(model, device)
    inputs = tok(text, return_tensors="pt").to(device)
    torch.manual_seed(0)  # VITS es estocástico; semilla fija para evaluaciones reproducibles
    with torch.inference_mode():
        wav = m(**inputs).waveform[0].float().cpu().numpy()
    return wav, m.config.sampling_rate


def _piper(text, model, device):
    # ponytail: onnxruntime en CPU; use_cuda requiere onnxruntime-gpu, que no instalamos.
    voice = _load_piper(model, False)
    chunks = list(voice.synthesize(text))
    audio = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(0, np.float32)
    return audio.astype(np.float32), voice.config.sample_rate


def speak(text: str, model: str = DEFAULT_MODEL, device: str | None = None) -> tuple[np.ndarray, int]:
    text = normalize(text)
    if not text:
        raise ValueError("texto vacío")
    if backend_of(model) == "piper":
        return _piper(text, model, device)
    return _mms(text, model, pick_device(device))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tts.py -q && uv run pytest -q`
Expected: 3 passed; whole fast suite passes. Then `uv run pytest -q -m slow -k roundtrip` (MMS path unchanged) on the local 3060.
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/guanani/tts.py tests/test_tts.py
git commit -m "feat: backend Piper (ONNX) en tts.speak

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Piper training environment and launcher

**Files:**
- Create: `scripts/piper_setup.sh`
- Create: `scripts/piper_train.sh`
- Modify: `.gitignore` (`envs/piper/`)
- Modify: `pyproject.toml` (nothing) — the Piper environment is not a uv project, it is a venv with the editable clone.
- Test: smoke run on the server (Step 3).

**Interfaces:**
- Consumes: `data/train/tts/<data>/{wavs/, metadata.csv}` (Task 1).
- Produces: `envs/piper/.venv` with `piper.train` importable and `monotonic_align` compiled; `scripts/piper_train.sh <name> <data> espeak|text [--resume] [extra LightningCLI args]` → `runs/<name>/lightning_logs/version_0/checkpoints/*.ckpt` and `runs/<name>/config.json`.

- [ ] **Step 1: Setup script**

`scripts/piper_setup.sh`:

```bash
#!/usr/bin/env bash
# Crea envs/piper: clon fijado de piper1-gpl (GPL-3.0, solo entrenamiento en servidor) + venv + monotonic_align.
set -euo pipefail
cd "$(dirname "$0")/.."
SHA=efffbfb226bfb511ebbcf55d0cecd8b35a89743d
mkdir -p envs/piper
[ -d envs/piper/repo ] || git clone -q https://github.com/OHF-Voice/piper1-gpl envs/piper/repo
git -C envs/piper/repo checkout -q $SHA
[ -d envs/piper/.venv ] || uv venv -q -p 3.12 envs/piper/.venv
uv pip install -q -p envs/piper/.venv -e "envs/piper/repo[train]"
(cd envs/piper/repo && source ../.venv/bin/activate && ./build_monotonic_align.sh && python setup.py build_ext --inplace -q)
envs/piper/.venv/bin/python -c "import torch, piper.train; print('piper ok', torch.__version__, torch.cuda.is_available())"
```

`scripts/piper_train.sh`:

```bash
#!/usr/bin/env bash
# Uso: scripts/piper_train.sh <name> <data> espeak|text [--resume] [args extra de LightningCLI]
# Entrena Piper *medium* DESDE CERO (sin --ckpt_path: las voces publicadas heredan licencias de investigación).
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; data=$(realpath -e "data/train/tts/$2"); kind=$3; shift 3
resume=""
if [ "${1:-}" = "--resume" ]; then shift; resume=1; fi
run=runs/$name
if [ -z "$resume" ]; then rm -rf "$run"; fi
mkdir -p "$run"; run=$(realpath "$run")
case $kind in
  espeak) ptype=() ;;
  text) ptype=(--data.phoneme_type text) ;;
  *) echo "tipo de entrada no válido: $kind (espeak|text)" >&2; exit 2 ;;
esac
ckpt=()
if [ -n "$resume" ]; then
  last=$(ls "$run"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | tail -1)
  [ -n "$last" ] && ckpt=(--ckpt_path "$last")
fi
exec envs/piper/.venv/bin/python -m piper.train fit \
  --data.voice_name "$name" \
  --data.csv_path "$data/metadata.csv" \
  --data.audio_dir "$data/wavs" \
  --data.espeak_voice gn \
  "${ptype[@]}" \
  --data.cache_dir "$run/cache" \
  --data.config_path "$run/config.json" \
  --data.validation_split 0.03 \
  --data.batch_size 32 \
  --data.num_workers 4 \
  --model.sample_rate 22050 \
  --trainer.default_root_dir "$run" \
  --trainer.max_time 00:22:00:00 \
  "${ckpt[@]}" "$@"
```

Note: `text` mode passes the normalized Guarani text as codepoints (Piper applies NFD itself), so `metadata.csv` is the same for both modes.

Append to `.gitignore`: `envs/piper/`.

- [ ] **Step 2: Setup on the server**

Run (after rsync): `ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && bash scripts/piper_setup.sh 2>&1 | tail -3'`
Expected: `piper ok <torch version> True`. If `build_monotonic_align.sh` fails for missing `cythonize`, `uv pip install -p envs/piper/.venv cython` and rerun; ledger a Ruling. If CUDA is False on Blackwell, reinstall torch from the cu128 index into that venv (`uv pip install -p envs/piper/.venv torch --index-url https://download.pytorch.org/whl/cu128`); ledger a Ruling.

- [ ] **Step 3: Smoke run on the server (20 clips, few steps)**

```bash
ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && \
  uv run guanani train tts-prep --name smoke --max-cer 1.0 > /dev/null && \
  head -20 data/train/tts/smoke/metadata.csv > /tmp/m && mv /tmp/m data/train/tts/smoke/metadata.csv && \
  timeout 1200 bash scripts/piper_train.sh smoke smoke text --trainer.max_steps 20 --trainer.val_check_interval 10 --data.validation_split 0.1 > smoke.log 2>&1; \
  echo exit $?; find runs/smoke -name "*.ckpt" | head; ls runs/smoke/config.json'
```

Expected: `exit 0`, at least `last.ckpt`, and `runs/smoke/config.json`. (Full `tts-prep` here runs W2 on all ~5,500 clips — a few minutes; `--max-cer 1.0` keeps everything that passes the text/audio filters.) Then check eSpeak mode too: `timeout 1200 bash scripts/piper_train.sh smoke-es smoke espeak --trainer.max_steps 20 --data.validation_split 0.1` → `exit 0`. If eSpeak rejects `gn`, ledger a Ruling and drop T1 (text-only T2). Keep `runs/smoke` for Task 4's smoke; delete `runs/smoke-es`.

- [ ] **Step 4: Commit**

```bash
git add scripts/piper_setup.sh scripts/piper_train.sh .gitignore
git commit -m "feat: entorno y lanzador de entrenamiento de Piper desde cero

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Export to ONNX and pick the best checkpoint

**Files:**
- Create: `src/guanani/train/piper.py`
- Modify: `src/guanani/cli.py` (`train tts-pick`)
- Test: `tests/test_piper.py`

**Interfaces:**
- Consumes: `tts.speak`, `stt.transcribe`, `stt.SR`, `eval.score_asr`, `data.load_flores("dev")`, `runs/<name>/lightning_logs/version_*/checkpoints/*.ckpt`, `runs/<name>/config.json`.
- Produces: `piper.PIPER_ENV: Path` (env var `GUANANI_PIPER_ENV`); `piper.checkpoints(run_dir: Path) -> list[Path]` (sorted, unique); `piper.export(ckpt: Path, config: Path, out_dir: Path, name: str) -> Path`; `piper.roundtrip_cer(voice_dir: str, texts: list[str], asr_model: str) -> float`; `piper.pick_best(name, run_dir, asr_model="models/w2", dev_limit=100, models_dir=Path("models")) -> dict` → `models/<name>/{<name>.onnx, <name>.onnx.json, train_meta.json}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_piper.py
import json

import pytest

from guanani.train import piper


def _run(tmp_path, names):
    ck = tmp_path / "run" / "lightning_logs" / "version_0" / "checkpoints"
    ck.mkdir(parents=True)
    for n in names:
        (ck / n).write_text(n)
    (tmp_path / "run" / "config.json").write_text(json.dumps({"audio": {"sample_rate": 22050}}))
    return tmp_path / "run"


def test_checkpoints_lists_unique_sorted(tmp_path):
    run = _run(tmp_path, ["epoch=9-val_mel=0.5.ckpt", "last.ckpt", "epoch=3-val_mos=3.1.ckpt"])
    assert [p.name for p in piper.checkpoints(run)] == ["epoch=3-val_mos=3.1.ckpt", "epoch=9-val_mel=0.5.ckpt", "last.ckpt"]


def test_pick_best_without_checkpoints_raises(tmp_path):
    run = _run(tmp_path, [])
    with pytest.raises(ValueError, match="checkpoint"):
        piper.pick_best("t1", run, models_dir=tmp_path / "models")


def test_pick_best_prefers_lowest_cer(tmp_path, monkeypatch):
    run = _run(tmp_path, ["a.ckpt", "b.ckpt", "last.ckpt"])
    cers = {"a": 0.20, "b": 0.08, "last": 0.12}

    def fake_export(ckpt, config, out_dir, name):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.onnx").write_text(ckpt.stem)
        (out_dir / f"{name}.onnx.json").write_text(config.read_text())
        return out_dir / f"{name}.onnx"

    monkeypatch.setattr(piper, "export", fake_export)
    monkeypatch.setattr(piper, "dev_texts", lambda limit: ["x"])
    monkeypatch.setattr(piper, "roundtrip_cer", lambda voice_dir, texts, asr: cers[piper.voice_onnx(voice_dir).read_text()])
    meta = piper.pick_best("t1", run, models_dir=tmp_path / "models")
    out = tmp_path / "models" / "t1"
    assert meta["best"] == "b.ckpt" and meta["best_cer"] == 0.08
    assert (out / "t1.onnx").read_text() == "b" and (out / "t1.onnx.json").exists()
    assert json.loads((out / "train_meta.json").read_text())["history"]["a.ckpt"] == 0.20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_piper.py -q`
Expected: FAIL (`ImportError: cannot import name 'piper'`).

- [ ] **Step 3: Implement**

```python
# src/guanani/train/piper.py
"""Piper: exporta checkpoints a ONNX (en envs/piper) y elige el mejor por CER de ida y vuelta en FLORES dev."""

import json
import os
import shutil
import subprocess
from pathlib import Path

PIPER_ENV = Path(os.environ.get("GUANANI_PIPER_ENV", Path(__file__).resolve().parents[3] / "envs" / "piper"))


def checkpoints(run_dir: Path) -> list[Path]:
    return sorted(set(run_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt")), key=lambda p: p.name)


def voice_onnx(voice_dir: str) -> Path:
    return next(Path(voice_dir).glob("*.onnx"))


def export(ckpt: Path, config: Path, out_dir: Path, name: str) -> Path:
    python = PIPER_ENV / ".venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"Falta el entorno de Piper en {PIPER_ENV}: ejecuta `bash scripts/piper_setup.sh`")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx = out_dir / f"{name}.onnx"
    subprocess.run([str(python), "-m", "piper.train.export_onnx", "--checkpoint", str(ckpt), "--output-file", str(onnx)],
                   check=True)
    shutil.copy(config, out_dir / f"{name}.onnx.json")
    return onnx


def dev_texts(limit: int) -> list[str]:
    from guanani import data
    from guanani.normalize import normalize

    return [normalize(r["gn"]) for r in data.load_flores("dev")][:limit]


def roundtrip_cer(voice_dir: str, texts: list[str], asr_model: str) -> float:
    import librosa

    from guanani.eval import score_asr
    from guanani.stt import SR, transcribe
    from guanani.tts import _load_piper, speak

    wavs = []
    for t in texts:
        a, sr = speak(t, model=voice_dir)
        wavs.append(librosa.resample(a, orig_sr=sr, target_sr=SR))
    _load_piper.cache_clear()  # cada checkpoint exporta al mismo nombre de archivo
    return score_asr(transcribe(wavs, model=asr_model), texts)["cer"]


def pick_best(name: str, run_dir: Path, asr_model: str = "models/w2", dev_limit: int = 100,
              models_dir: Path = Path("models")) -> dict:
    ckpts = checkpoints(run_dir)
    if not ckpts:
        raise ValueError(f"no hay checkpoints en {run_dir}/lightning_logs/version_*/checkpoints")
    texts = dev_texts(dev_limit)
    tmp = models_dir / f".{name}_eval"
    history = {}
    for ck in ckpts:
        voice = tmp / ck.stem
        export(ck, run_dir / "config.json", voice, name)
        history[ck.name] = round(roundtrip_cer(str(voice), texts, asr_model), 4)
        print(f"[dev] {ck.name}: CER {history[ck.name]}", flush=True)
    best = min(history, key=history.get)
    out = models_dir / name
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(tmp / Path(best).stem, out)
    shutil.rmtree(tmp, ignore_errors=True)
    meta = {"run_dir": str(run_dir), "best": best, "best_cer": history[best], "asr_model": asr_model,
            "dev": len(texts), "history": history}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
```

CLI (inside `train` subparsers):

```python
    tk = tsub.add_parser("tts-pick", help="exporta los checkpoints de Piper y deja el mejor en models/<name>")
    tk.add_argument("--name", required=True)
    tk.add_argument("--run-dir", required=True)
    tk.add_argument("--asr-model", default="models/w2")
    tk.add_argument("--dev-limit", type=int, default=100)
```

Branch:

```python
            elif a.train_cmd == "tts-pick":
                from pathlib import Path

                from guanani.train.piper import pick_best

                print(json.dumps(pick_best(a.name, Path(a.run_dir), a.asr_model, a.dev_limit), indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_piper.py -q && uv run pytest -q`
Expected: 3 passed; suite passes.

- [ ] **Step 5: Smoke on the server + slow test**

```bash
ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && \
  uv run guanani train tts-pick --name smoke --run-dir runs/smoke --dev-limit 5 2>&1 | grep -v -i warn | tail -8 && \
  ls models/smoke && uv run guanani speak --model models/smoke "Mba'\''éichapa" -o /tmp/s.wav && python3 -c "import soundfile as sf; print(sf.info(\"/tmp/s.wav\"))"'
```

Expected: JSON with `best` and `history`; `models/smoke/{smoke.onnx, smoke.onnx.json, train_meta.json}`; a WAV at 22,050 Hz (noise after 20 steps is fine).

Append to `tests/test_models_slow.py`:

```python
@pytest.mark.slow
def test_piper_voice_speaks():
    from pathlib import Path

    from guanani import tts

    voice = Path("models/smoke")
    if not voice.exists():
        pytest.skip("sin voz de humo (Task 4 Step 5 en el servidor)")
    audio, sr = tts.speak("Mba'éichapa", model=str(voice))
    assert sr == 22050 and len(audio) > sr // 10
```

Run on the server: `uv run pytest -q -m slow -k piper_voice` → 1 passed. Then `rm -rf runs/smoke models/smoke data/train/tts/smoke`.

- [ ] **Step 6: Commit**

```bash
git add src/guanani/train/piper.py src/guanani/cli.py tests/test_piper.py tests/test_models_slow.py
git commit -m "feat: exportación de Piper a ONNX y elección del mejor checkpoint por CER de ida y vuelta

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: TTS evaluation with a selectable recogniser and UTMOS

**Files:**
- Modify: `src/guanani/eval.py` (`_eval_tts`, `run`, `score_utmos`)
- Modify: `src/guanani/cli.py` (`eval --asr-model`, `--no-utmos`)
- Test: `tests/test_report.py` or `tests/test_metrics.py` (unit), slow UTMOS check

**Interfaces:**
- Consumes: `tts.speak`, `stt.transcribe`.
- Produces: `eval.score_utmos(wavs: list[np.ndarray], sr: int) -> float` (mean; `utmos22_strong` via `torch.hub`); `eval.run(..., asr_model: str | None = None, utmos: bool = True)`; TTS metrics `roundtrip_cer`, `roundtrip_wer`, `utmos`, and `asr_model` in the report.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py`:

```python
def test_eval_tts_uses_given_asr_model_and_utmos(tmp_path, monkeypatch):
    import numpy as np

    from guanani import eval as ev
    from guanani import stt, tts

    seen = {}
    monkeypatch.setattr(tts, "speak", lambda text, model, device: (np.zeros(16000, np.float32), 16000))
    monkeypatch.setattr(stt, "transcribe", lambda wavs, model, device: seen.setdefault("asr", model) and ["a"] * len(wavs))
    monkeypatch.setattr(ev, "score_utmos", lambda wavs, sr: 3.5)
    rep = ev._eval_tts([{"gn": "a"}], "voz", "cpu", tmp_path / "audio", asr_model="models/w2", utmos=True)
    assert seen["asr"] == "models/w2" and rep["asr_model"] == "models/w2"
    assert rep["metrics"]["utmos"] == 3.5 and rep["metrics"]["roundtrip_cer"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_metrics.py -q -k utmos`
Expected: FAIL (`TypeError: _eval_tts() got an unexpected keyword argument 'asr_model'`).

- [ ] **Step 3: Implement**

In `src/guanani/eval.py` add:

```python
def score_utmos(wavs: list, sr: int) -> float:
    import torch

    # ponytail: UTMOS22 (SpeechMOS vía torch.hub, descarga el código de GitHub); solo evaluación.
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    with torch.inference_mode():
        scores = [float(predictor(torch.from_numpy(w).float().unsqueeze(0), sr)) for w in wavs]
    return round(sum(scores) / len(scores), 3)
```

Change `_eval_tts` signature to `def _eval_tts(rows, model, device, audio_dir: Path, asr_model: str | None = None, utmos: bool = True):`, set `asr = asr_model or ASR_MODEL`, call `transcribe(wavs, model=asr, device=device)`, and build metrics as:

```python
    metrics = {f"roundtrip_{k}": v for k, v in score_asr(hyps, texts).items()}
    if utmos:
        metrics["utmos"] = score_utmos(wavs, SR)
```

and add `"asr_model": asr` to the returned dict (replacing the old `"asr_model": ASR_MODEL`). In `run(...)` add parameters `asr_model: str | None = None, utmos: bool = True` and pass them to `_eval_tts`.

CLI `eval` parser: `e.add_argument("--asr-model", help="reconocedor de ida y vuelta para tts")` and `e.add_argument("--no-utmos", action="store_true")`; in the eval branch pass `asr_model=a.asr_model, utmos=not a.no_utmos` to `run`.

The report row in `SUMMARY.md` already prints every non-dict metric, so `utmos=…` appears automatically; add the ASR model to the row's model column when it is not the default: in `run`, for tts set `report["model"] = f"{model} (asr {asr_model})"` if `asr_model` is given.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/guanani/eval.py src/guanani/cli.py tests/test_metrics.py
git commit -m "feat: eval tts con reconocedor elegible (--asr-model) y UTMOS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Experiments T0–T2 (T3 if applicable) and conclusions

**Files:**
- Modify: `results/SUMMARY.md`
- Modify: memory `guanani-project-decisions.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `models/t1`, `models/t2` (ONNX) on the server and locally; rows and conclusions.

- [ ] **Step 1: Prepare the voice on the server**

`nvidia-smi` must show the GPU free (see Global Constraints). Then:
`ssh user@gpu-server 'cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH && uv run guanani train tts-prep --name v1'`
Expected: `stats.json` with `hablante`, `aceptados`, `horas`. **If `horas` < 3**, apply plan B (T3) instead of Step 3: prepare every speaker with ≥ 1 h (`--speaker <id>` per speaker, ledger the list), write a multi-speaker `metadata.csv` (`file.wav|speaker|text`), train with `--model.num_speakers N`, then fine-tune on `v1` with `--ckpt_path` of that run; ledger a Ruling.

- [ ] **Step 2: T0 (baseline with W2)**

`uv run guanani eval tts flores --limit 200 --asr-model models/w2` (MMS-TTS, GPU) and `uv run guanani --device cpu eval tts flores --limit 20 --asr-model models/w2 --no-utmos` (CPU RTF).
Expected: two rows in the server `results/SUMMARY.md`.

- [ ] **Step 3: T1 and T2 in parallel**

```bash
ssh user@gpu-server 'cd ~/Guanani && cat > chain_tts.sh <<'"'"'EOF'"'"'
cd ~/Guanani && export PATH=$HOME/.local/bin:$PATH
bash scripts/piper_train.sh t1 v1 espeak > t1.log 2>&1 &
bash scripts/piper_train.sh t2 v1 text > t2.log 2>&1 &
wait
for n in t1 t2; do
  uv run guanani train tts-pick --name $n --run-dir runs/$n >> $n.log 2>&1
  uv run guanani eval tts flores --limit 200 --model models/$n --asr-model models/w2 >> $n.log 2>&1
  uv run guanani --device cpu eval tts flores --limit 20 --model models/$n --asr-model models/w2 --no-utmos >> $n.log 2>&1
done
echo FIN
EOF
setsid nohup bash chain_tts.sh > chain_tts.log 2>&1 < /dev/null & disown'
```

Expected after ~22 h: `FIN`; `models/t1`, `models/t2`; four rows. If two trainings don't fit in 24 GB together (OOM in either log), run them sequentially (`--trainer.max_time 00:11:00:00` each) and ledger a Ruling.

- [ ] **Step 4: Results and conclusions**

Bring only the new `| tts |` rows from the server's `results/SUMMARY.md` and their JSON, plus `results/audio/<date>_models-t1*` / `t2*` samples. Copy the chosen voice locally: `rsync -a user@gpu-server:~/Guanani/models/<best> models/`. Add "Conclusions (subproject 3 — commercial TTS, <date>)" above subproject 2 in `results/SUMMARY.md`: table T0/T1/T2 (license, roundtrip CER/WER with W2, UTMOS, RTF CPU), threshold check, hours and discards from `stats.json`, T1 vs T2 (eSpeak vs letters), the donor-voice caveat, and whether plan B was needed. Check locally: `uv run guanani speak --model models/<best> "Mba'éichapa, mba'éichapa reiko?" -o /tmp/x.wav`.

- [ ] **Step 5: Commit and memory**

Run: `uv run pytest -q`.

```bash
git add results/
git commit -m "docs: resultados y conclusiones del subproyecto 3 (TTS comercial)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Update `~/.claude/projects/-home-josemi-Documents-Personal-Ideas-Guanani/memory/guanani-project-decisions.md` with a "3 TTS" line (best voice, metrics, where it lives, plan B used or not).
