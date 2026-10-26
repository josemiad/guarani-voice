"""Métricas e informes de evaluación."""

import json
import re
import time
from datetime import datetime
from pathlib import Path

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


def score_utmos(wavs: list, sr: int) -> float:
    import torch

    # ponytail: UTMOS22 (SpeechMOS vía torch.hub, descarga el código de GitHub); solo evaluación.
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    with torch.inference_mode():
        scores = [float(predictor(torch.from_numpy(w).float().unsqueeze(0), sr)) for w in wavs]
    return round(sum(scores) / len(scores), 3)


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


SAMPLES = 20
AUDIO_SAMPLES = 10
COMPATIBLE = {
    "nmt": {"flores", "jojajovai", "jojajovai-anotado"},
    "stt": {"cv", "cv-dev"},
    "tts": {"flores", "jojajovai", "jojajovai-anotado", "cv"},
}
HEADER = (
    "| date | component | model | dataset | direction | n | metrics | latency | device |\n"
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
        summary.write_text("# Results\n\n" + HEADER, encoding="utf-8")
    metrics = " ".join(f"{k}={v}" for k, v in report["metrics"].items() if not isinstance(v, dict))
    lat = report.get("latency_s_per_item")
    latency = f"{lat} s/sentence" if lat is not None else f"RTF {report.get('rtf')}"
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
        # NLLB-1.3B y COMET no caben juntos en una GPU compartida de 12 GB: se libera NLLB antes.
        import gc

        import torch

        from guanani import nmt

        nmt._load.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()
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


def _eval_tts(rows, model, device, audio_dir: Path, asr_model: str | None = None, utmos: bool = True):
    import librosa
    import numpy as np
    import soundfile as sf

    from guanani.stt import DEFAULT_MODEL as ASR_MODEL
    from guanani.stt import SR, transcribe
    from guanani.tts import speak

    texts = [t for t in (normalize(r["gn"]) for r in rows) if t]
    wavs, rejected = [], set()
    t = time.perf_counter()
    for i, text in enumerate(texts):
        try:
            a, sr = speak(text, model=model, device=device)
        except ValueError:
            # La voz rechaza la frase (p. ej. cifras): cuenta como fallo para comparar sobre las mismas frases.
            rejected.add(i)
            a, sr = np.zeros(SR // 10, np.float32), SR
        wavs.append(a if sr == SR else librosa.resample(a, orig_sr=sr, target_sr=SR))
    secs = time.perf_counter() - t
    duration = sum(len(w) for w in wavs) / SR
    asr = asr_model or ASR_MODEL
    hyps = transcribe(wavs, model=asr, device=device)
    hyps = ["" if i in rejected else h for i, h in enumerate(hyps)]
    metrics = {f"roundtrip_{k}": v for k, v in score_asr(hyps, texts).items()}
    # Las voces Piper no leen cifras: CER también sobre las frases sin dígitos, comparable entre voces.
    plain = [i for i, x in enumerate(texts) if not re.search(r"\d", x)]
    if plain and len(plain) < len(texts):
        metrics["roundtrip_cer_sin_cifras"] = score_asr([hyps[i] for i in plain], [texts[i] for i in plain])["cer"]
        metrics["n_sin_cifras"] = len(plain)
    if utmos:
        metrics["utmos"] = score_utmos([w for i, w in enumerate(wavs) if i not in rejected], SR)
    audio_dir.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(wavs[:AUDIO_SAMPLES]):
        sf.write(audio_dir / f"{i:02d}.wav", w, SR)
    return {
        "n": len(texts),
        "rejected": len(rejected),
        "asr_model": asr,
        "audio_dir": str(audio_dir),
        "metrics": metrics,
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
    asr_model: str | None = None,
    utmos: bool = True,
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
        report = _eval_tts(rows, model, device, Path(results_dir) / "audio" / f"{date}_{_slug(model)}", asr_model, utmos)
        if asr_model:
            model = f"{model} (asr {asr_model})"
    report |= {"date": date, "component": component, "model": model, "dataset": dataset, "device": device, "limit": limit}
    return write_report(report, Path(results_dir))
