import csv
import json

import numpy as np
import pytest
import soundfile as sf

from guanani.train import tts_data

HEAD = "client_id\tpath\tsentence\n"


def _cv(tmp_path, monkeypatch, rows_by_split):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    d = tmp_path / "common_voice"
    (d / "clips").mkdir(parents=True)
    for split, rows in rows_by_split.items():
        (d / f"{split}.tsv").write_text(HEAD + "".join(f"{s}\t{p}\t{t}\n" for s, p, t in rows), encoding="utf-8")
    f = tmp_path / "flores"
    f.mkdir()
    for split in ("dev", "devtest"):
        for code in ("gug_Latn", "spa_Latn"):
            (f / f"{split}.{code}.jsonl").write_text(json.dumps({"id": 1, "text": f"Frase de {split}"}), encoding="utf-8")
    return d / "clips"


def _wav(path, seconds, amp=0.1, sr=48000):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)


def test_text_reason():
    keys = {"frase de dev"}
    assert tts_data.text_reason("Che róga", keys) is None
    assert tts_data.text_reason("   ", keys) == "vacia"
    assert tts_data.text_reason("Ary 2024-pe", keys) == "cifras"
    assert tts_data.text_reason("Frase de dev.", keys) == "flores"


def test_audio_reason():
    sr = tts_data.SR
    tone = lambda s, a=0.1: (a * np.sin(np.arange(int(s * sr)) / 7)).astype(np.float32)
    assert tts_data.audio_reason(tone(3), sr) is None
    assert tts_data.audio_reason(tone(0.5), sr) == "corto"
    assert tts_data.audio_reason(tone(16), sr) == "largo"
    clipped = tone(3, a=1.5).clip(-1, 1)
    assert tts_data.audio_reason(clipped, sr) == "saturado"


def test_to_voice_audio_resamples_and_normalizes(tmp_path):
    _wav(tmp_path / "a.wav", 2.0, amp=0.5)
    a = tts_data.to_voice_audio(str(tmp_path / "a.wav"))
    rms_db = 20 * np.log10(np.sqrt(np.mean(a**2)))
    assert abs(len(a) - 2 * tts_data.SR) < 50
    assert abs(rms_db - (-20)) < 0.5


def test_top_speaker_counts_all_splits(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, {"test": [("a", "1.mp3", "x")], "dev": [("b", "2.mp3", "y")],
                                "train": [("b", "3.mp3", "z")], "other": [("c", "4.mp3", "w")]})
    assert tts_data.top_speaker() == "b"


def test_prepare_voice_filters_and_writes_piper_metadata(tmp_path, monkeypatch):
    clips = _cv(tmp_path, monkeypatch, {
        "test": [], "dev": [], "train": [
            ("v", "ok.wav", "Che róga porã"),
            ("v", "quoted.wav", '"Che róga" he\'i'),
            ("v", "misread.wav", "Mba'éichapa"),
            ("v", "broken.wav", "Ndaipóri"),
            ("v", "digits.wav", "Ary 1990"),
            ("w", "other_speaker.wav", "Ambue"),
        ], "other": []})
    for name in ("ok", "quoted", "misread", "digits", "other_speaker"):
        _wav(clips / f"{name}.wav", 3.0)
    (clips / "broken.wav").write_bytes(b"not audio")
    fake = {"Che róga porã": "che róga porã", '"Che róga" he\'i': "che róga he'i",
            "Mba'éichapa": "otra cosa totalmente distinta"}
    out = tts_data.prepare_voice("v1", speaker="v", out_root=tmp_path / "tts",
                                 transcribe=lambda texts_audios: [fake[t] for t, _ in texts_audios])
    rows = list(csv.reader((tmp_path / "tts" / "v1" / "metadata.csv").open(encoding="utf-8"), delimiter="|"))
    # Piper lee el CSV con csv.reader: las comillas del texto deben ir escapadas o juntan líneas.
    assert rows == [["ok.wav", "Che róga porã"], ["quoted.wav", '"Che róga" he\'i']]
    assert (tmp_path / "tts" / "v1" / "wavs" / "ok.wav").exists()
    from guanani.tts import piper_text

    text_rows = list(csv.reader((tmp_path / "tts" / "v1" / "metadata_text.csv").open(encoding="utf-8"), delimiter="|"))
    assert text_rows[0] == ["ok.wav", piper_text("Che róga porã")]
    assert sf.info(tmp_path / "tts" / "v1" / "wavs" / "ok.wav").samplerate == tts_data.SR
    assert out["descartes"] == {"cifras": 1, "ilegible": 1, "asr": 1}
    assert out["aceptados"] == 2 and out["horas"] > 0
    assert json.loads((tmp_path / "tts" / "v1" / "stats.json").read_text())["aceptados"] == 2


def test_prepare_voice_unknown_speaker_raises(tmp_path, monkeypatch):
    _cv(tmp_path, monkeypatch, {"test": [], "dev": [], "train": [("v", "a.wav", "x")], "other": []})
    with pytest.raises(ValueError, match="nadie"):
        tts_data.prepare_voice("v1", speaker="nadie", out_root=tmp_path / "tts", transcribe=lambda x: [])
