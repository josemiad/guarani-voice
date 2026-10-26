"""Síntesis de voz en guaraní: Piper (ONNX, voces propias) y MMS-TTS (referencia no comercial)."""

import re
import unicodedata
from functools import cache
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from guanani.device import pick_device
from guanani.normalize import normalize

DEFAULT_MODEL = "facebook/mms-tts-grn"


# Voces Piper en modo letras: la tabla de símbolos de Piper es IPA (sin mayúsculas ni acento combinante).
# El acento agudo marca la sílaba tónica en guaraní → marca de acento IPA; U+0342 es una tilde mal tecleada.
_PIPER_TEXT_MAP = {"\u0301": "ˈ", "\u0342": "\u0303"}
_PIPER_TEXT_DROP = set("¡¿“”«»")


def piper_text(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.lower())
    return "".join(_PIPER_TEXT_MAP.get(c, c) for c in nfd if c not in _PIPER_TEXT_DROP)


def voice_file(model: str) -> Path:
    return next(Path(model).glob("*.onnx"))


def backend_of(model: str) -> str:
    p = Path(model)
    if p.is_dir() and any(p.glob("*.onnx")):
        onnx = voice_file(model)
        if not onnx.with_suffix(".onnx.json").exists():
            raise ValueError(f"falta {onnx.name}.json junto a {onnx}: exporta la voz con `guanani train tts-pick`")
        return "piper"
    if p.exists() or model.startswith(("models/", "./", "/")):
        raise ValueError(f"no hay una voz Piper en {model}: créala con `guanani train tts-pick` o copia models/<voz>")
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
    sr = voice.config.sample_rate
    if getattr(voice.config.phoneme_type, "value", None) != "text":
        chunks = list(voice.synthesize(text))  # eSpeak ya trocea por frases
        if not chunks:
            raise ValueError("texto vacío tras la fonetización")
        return np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32), sr
    if re.search(r"\d", text):
        # La voz se entrenó sin frases con cifras: sus símbolos no están aprendidos y sonarían a ruido.
        raise ValueError("las cifras no están soportadas: escríbelas en letras")
    # En modo letras Piper no trocea: una frase por pasada, como en el entrenamiento (clips ≤ 15 s).
    sentences = [piper_text(t) for t in re.split(r"(?<=[.!?;:])\s+", text)]
    sentences = [t for t in sentences if t.strip()]
    if not sentences:
        raise ValueError("texto vacío tras convertirlo a símbolos de la voz")
    pause = np.zeros(int(0.2 * sr), np.float32)
    parts = []
    for t in sentences:
        parts += [c.audio_float_array for c in voice.synthesize(t)] + [pause]
    return np.concatenate(parts[:-1]).astype(np.float32), sr


def speak(text: str, model: str = DEFAULT_MODEL, device: str | None = None) -> tuple[np.ndarray, int]:
    text = normalize(text)
    if not text:
        raise ValueError("texto vacío")
    if backend_of(model) == "piper":
        return _piper(text, model, device)
    return _mms(text, model, pick_device(device))
