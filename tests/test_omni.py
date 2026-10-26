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
    for s in (500, 1000):  # fairseq2 escribe en <run>/ws_<rank>.<hash>/checkpoints/step_N/model
        (run / "ws_1.abc" / "checkpoints" / f"step_{s}" / "model").mkdir(parents=True)
        (run / "ws_1.abc" / "checkpoints" / f"step_{s}" / "model" / "w.pt").write_text(str(s))
    cers = {"omniASR_CTC_1B_v2": 0.08, "500": 0.05, "1000": 0.06}
    monkeypatch.setattr(omni, "dev_cer", lambda model, rows: cers[model if model.startswith("omniASR") else model.rsplit("_step_", 1)[1]])
    monkeypatch.setattr(omni, "dev_rows", lambda limit: [])
    out = omni.pick_best("s1", run, "1b_v2", models_dir=tmp_path / "models")
    assert out["best_step"] == 500 and out["history"]["0"] == 0.08
    # El punto elegido se copia dentro de models/<name>: el modelo no depende de runs/.
    out = tmp_path / "models" / "s1"
    assert (out / "checkpoint" / "w.pt").read_text() == "500"
    assert "checkpoint: checkpoint\n" in (out / "card.yaml").read_text()  # relativa: models/<name> es portable
