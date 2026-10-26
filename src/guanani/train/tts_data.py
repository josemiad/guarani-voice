"""Voz para Piper: clips de un hablante de Common Voice, filtrados por W2, a 22.050 Hz y −20 dBFS."""

import csv
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from guanani import data
from guanani.normalize import normalize
from guanani.tts import piper_text
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
        kept.append([fname, text])
        seconds += len(audio) / SR
    buf = io.StringIO()
    # csv.writer escapa las comillas: Piper lee con csv.reader y un texto que empieza por " juntaría líneas.
    csv.writer(buf, delimiter="|", lineterminator="\n").writerows(kept)
    (out / "metadata.csv").write_text(buf.getvalue(), encoding="utf-8")
    # Modo letras de Piper: mismos clips con el texto ya en símbolos de su tabla (ver tts.piper_text).
    buf = io.StringIO()
    csv.writer(buf, delimiter="|", lineterminator="\n").writerows([f, piper_text(t)] for f, t in kept)
    (out / "metadata_text.csv").write_text(buf.getvalue(), encoding="utf-8")
    stats = {"hablante": speaker, "entrada": len(rows), "aceptados": len(kept), "horas": round(seconds / 3600, 3),
             "descartes": dict(drops), "asr_model": asr_model, "max_cer": max_cer}
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats
