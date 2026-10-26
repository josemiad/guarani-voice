"""Reconocimiento de voz en guaraní: MMS (referencia no comercial), Whisper y Omnilingual."""

import json
import os
import re
import subprocess
import tempfile
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
MAX_CHUNK_S = 10.0  # control de audios largos: 30 s → WER 38–42 %, 10 s → 30–31 % (W2, S2b)
OMNI_ENV = Path(os.environ.get("GUANANI_OMNI_ENV", Path(__file__).resolve().parents[2] / "envs" / "omnilingual"))
OMNI_LANG = "grn_Latn"  # código con el que Omnilingual entrenó guaraní (38,6 h según su tabla por idioma)


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if audio.size == 0:
        raise ValueError(f"audio vacío: {path}")
    return audio


def resolve_model(model: str) -> str:
    # `guanani train stt` guarda los pesos en models/<name>/best_model; aceptar también models/<name>.
    best = Path(model) / "best_model"
    return str(best) if (best / "config.json").exists() else model


def backend_of(model: str) -> str:
    p = Path(resolve_model(model))
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


def _omnilingual(audios, model, batch_size):
    # fairseq2 exige otro torch: Omnilingual vive en su propio entorno y se llama por subprocess.
    python = OMNI_ENV / ".venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError(f"Falta el entorno de Omnilingual en {OMNI_ENV}: ejecuta `uv sync --project envs/omnilingual`")
    p = Path(model)
    card, asset_dir, card_text = model, None, None
    if (p / "card.yaml").exists():
        # El nombre sale de la tarjeta, no del directorio: así el modelo se puede copiar o renombrar.
        text = (p / "card.yaml").read_text(encoding="utf-8")
        card, asset_dir = re.search(r"^name:\s*(\S+)", text, re.M).group(1), str(p)
        ckpt = re.search(r"^checkpoint:\s*(\S+)", text, re.M)
        if ckpt and "://" not in ckpt.group(1) and not Path(ckpt.group(1)).is_absolute():
            # Ruta relativa a la tarjeta (modelo portable): fairseq2 necesita una absoluta.
            card_text = text.replace(ckpt.group(0), f"checkpoint: {(p / ckpt.group(1)).resolve()}")
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.npz", Path(tmp) / "out.jsonl"
        if card_text:  # tarjeta con la ruta ya absoluta, en un directorio temporal
            asset_dir = str(Path(tmp) / "card")
            Path(asset_dir).mkdir()
            (Path(asset_dir) / "card.yaml").write_text(card_text, encoding="utf-8")
        np.savez(inp, *audios)
        cmd = [str(python), str(OMNI_ENV / "transcribe.py"), "--model", card, "--batch-size", str(batch_size)]
        if asset_dir:
            cmd += ["--asset-dir", asset_dir]
        if "_LLM_" in card:
            cmd += ["--lang", OMNI_LANG]
        subprocess.run(cmd + [str(inp), str(out)], check=True)
        return [json.loads(line)["text"] for line in out.read_text(encoding="utf-8").splitlines()]


def transcribe(audios: list[np.ndarray], model: str = DEFAULT_MODEL, device: str | None = None,
               batch_size: int = 16) -> list[str]:
    device = pick_device(device)
    model = resolve_model(model)
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
