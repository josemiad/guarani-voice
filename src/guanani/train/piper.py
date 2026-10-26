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
    script = Path(__file__).resolve().parents[3] / "scripts" / "piper_export.py"  # fuerza el exportador clásico
    subprocess.run([str(python), str(script), "--checkpoint", str(ckpt), "--output-file", str(onnx)], check=True)
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
