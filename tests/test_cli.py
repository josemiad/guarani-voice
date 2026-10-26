from guanani.cli import main


def test_eval_missing_data_exits_2_with_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    assert main(["eval", "nmt", "flores", "--no-comet"]) == 2
    assert "Falta" in capsys.readouterr().err


def test_eval_incompatible_dataset_exits_2(capsys):
    assert main(["eval", "stt", "flores"]) == 2
    assert "stt" in capsys.readouterr().err


def test_train_lora_missing_bt_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "parallel.jsonl").write_text('{"gn": "a", "es": "b", "source": "s"}\n', encoding="utf-8")
    assert main(["train", "lora", "--name", "x", "--data", "parallel,bt"]) == 2
    assert "backtranslate" in capsys.readouterr().err


def test_train_prepare_missing_data_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    assert main(["train", "prepare"]) == 2
    assert "Falta" in capsys.readouterr().err


def test_train_lora_missing_bt2_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "parallel.jsonl").write_text('{"gn": "a", "es": "b", "source": "s"}\n', encoding="utf-8")
    assert main(["train", "lora", "--name", "x", "--data", "parallel,bt2", "--upsample-parallel", "4", "--bt-tag"]) == 2
    assert "backtranslate" in capsys.readouterr().err


def test_train_nmt_is_the_new_name_and_lora_still_works(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "parallel.jsonl").write_text('{"gn": "a", "es": "b", "source": "s"}\n', encoding="utf-8")
    for cmd in ("nmt", "lora"):
        assert main(["train", cmd, "--name", "x", "--data", "parallel,bt3", "--full"]) == 2
        assert "backtranslate" in capsys.readouterr().err


def test_train_stt_rejects_non_whisper_base(capsys):
    from guanani.cli import main

    assert main(["train", "stt", "--name", "x", "--base", "facebook/mms-1b-all"]) == 2
    assert "openai/whisper-" in capsys.readouterr().err


def test_transcribe_long_audio_switches_to_chunking(monkeypatch, capsys):
    import numpy as np

    from guanani import stt
    from guanani.cli import main

    monkeypatch.setattr(stt, "transcribe", lambda audios, **kw: ["corto"])
    monkeypatch.setattr(stt, "transcribe_long", lambda audio, **kw: "largo")
    for seconds, expected in ((5, "corto"), (60, "largo")):
        monkeypatch.setattr(stt, "load_audio", lambda path, s=seconds: np.zeros(s * stt.SR, np.float32))
        assert main(["transcribe", "x.wav"]) == 0
        assert capsys.readouterr().out.strip() == expected
