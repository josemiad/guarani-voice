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


def test_export_runs_legacy_exporter_script_in_piper_env(tmp_path, monkeypatch):
    import sys

    env = tmp_path / "env"
    (env / ".venv" / "bin").mkdir(parents=True)
    py = env / ".venv" / "bin" / "python"
    # Python falso: registra los argumentos y crea el .onnx pedido.
    py.write_text(f'#!/bin/sh\necho "$@" > {tmp_path}/args\nexec "{sys.executable}" -c "import sys, pathlib; '
                  f'a = sys.argv[1:]; pathlib.Path(a[a.index(\'--output-file\') + 1]).write_text(\'onnx\')" "$@"\n')
    py.chmod(0o755)
    monkeypatch.setattr(piper, "PIPER_ENV", env)
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    onnx = piper.export(tmp_path / "a.ckpt", cfg, tmp_path / "out", "t1")
    args = (tmp_path / "args").read_text()
    assert "piper_export.py" in args and "--checkpoint" in args
    assert onnx.read_text() == "onnx" and (tmp_path / "out" / "t1.onnx.json").exists()


def test_export_missing_env_explains_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(piper, "PIPER_ENV", tmp_path / "nope")
    with pytest.raises(RuntimeError, match="piper_setup.sh"):
        piper.export(tmp_path / "a.ckpt", tmp_path / "c.json", tmp_path / "out", "t1")
