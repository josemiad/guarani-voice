import pytest

from guanani.train import stt_data

HEAD = "client_id\tpath\tsentence\n"


def _cv(tmp_path, monkeypatch, **splits):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    d = tmp_path / "common_voice"
    d.mkdir()
    for split, sentences in splits.items():
        rows = "".join(f"spk_{split}\t{split}_{i}.mp3\t{s}\n" for i, s in enumerate(sentences))
        (d / f"{split}.tsv").write_text(HEAD + rows, encoding="utf-8")


def test_load_stt_drops_sentences_seen_in_test(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, test=["Mba'éichapa reiko?"], dev=["z"], train=["mba’éichapa reiko", "Che areko peteĩ jagua."])
    rows, stats = stt_data.load_stt(["train"])
    assert [r["text"] for r in rows] == ["Che areko peteĩ jagua."]
    assert stats["train"] == {"in": 2, "vacia": 0, "contaminada": 1, "hablante_eval": 0, "aceptada": 1}


def test_load_stt_normalizes_text_and_keeps_split(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, test=["x"], dev=["Ñande  rekove  âĝuã"])
    rows, _ = stt_data.load_stt(["dev"])
    assert rows[0]["text"] == "Ñande rekove ãg̃uã"
    assert rows[0]["split"] == "dev"
    assert rows[0]["audio"].endswith("clips/dev_0.mp3")


def test_load_stt_counts_empty_sentences(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, test=["x"], dev=["z"], other=["   "])
    rows, stats = stt_data.load_stt(["other"])
    assert rows == [] and stats["other"]["vacia"] == 1


def test_load_stt_rejects_test_and_unknown_splits(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, test=["x"])
    with pytest.raises(ValueError, match="test"):
        stt_data.load_stt(["test"])
    with pytest.raises(ValueError, match="foo"):
        stt_data.load_stt(["foo"])


def test_cv_dev_dataset_is_registered(tmp_path, monkeypatch):
    from guanani import data, eval as ev

    _cv(tmp_path, monkeypatch, dev=["a"])
    assert data.DATASETS["cv-dev"]()[0]["gn"] == "a"
    assert "cv-dev" in ev.COMPATIBLE["stt"]


def test_load_stt_drops_clips_from_test_and_dev_speakers(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    d = tmp_path / "common_voice"
    d.mkdir()
    (d / "test.tsv").write_text("client_id\tpath\tsentence\nspkT\tt.mp3\tx\n", encoding="utf-8")
    (d / "dev.tsv").write_text("client_id\tpath\tsentence\nspkD\td.mp3\ty\n", encoding="utf-8")
    (d / "other.tsv").write_text(
        "client_id\tpath\tsentence\nspkT\to1.mp3\tuno\nspkD\to2.mp3\tdos\nspkO\to3.mp3\ttres\n", encoding="utf-8"
    )
    rows, stats = stt_data.load_stt(["other"])
    assert [r["text"] for r in rows] == ["tres"]
    assert stats["other"]["hablante_eval"] == 2
