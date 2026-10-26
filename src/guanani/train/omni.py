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
    # fairseq2 escribe en <run>/ws_<rank>.<hash>/checkpoints/step_N/model
    steps = sorted((int(p.name.removeprefix("step_")), p / "model") for p in run_dir.glob("ws_*/checkpoints/step_*"))
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
    if best_step == 0:
        ckpt = BASE_URLS[arch]
    else:
        # Se copia el punto elegido dentro de models/<name>: el modelo no depende de runs/.
        ckpt = out / "checkpoint"
        shutil.rmtree(ckpt, ignore_errors=True)
        shutil.copytree(dict(steps)[best_step], ckpt)
        ckpt = "checkpoint"  # relativa a la tarjeta: stt la resuelve al cargar
    write_card(out, ckpt, arch)
    meta = {"base": BASES[arch], "run_dir": str(run_dir), "best_step": best_step, "best_cer": history[str(best_step)], "history": history}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
