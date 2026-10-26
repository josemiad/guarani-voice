import pytest

from guanani import tts


def test_backend_of(tmp_path):
    v = tmp_path / "t1"
    v.mkdir()
    (v / "t1.onnx").write_bytes(b"x")
    (v / "t1.onnx.json").write_text("{}")
    assert tts.backend_of(str(v)) == "piper"
    assert tts.voice_file(str(v)) == v / "t1.onnx"
    assert tts.backend_of("facebook/mms-tts-grn") == "mms"


def test_backend_of_onnx_without_config_is_an_error(tmp_path):
    v = tmp_path / "t1"
    v.mkdir()
    (v / "t1.onnx").write_bytes(b"x")
    with pytest.raises(ValueError, match="onnx.json"):
        tts.backend_of(str(v))


def test_speak_empty_text_raises_before_loading_voice(tmp_path):
    v = tmp_path / "t1"
    v.mkdir()
    (v / "t1.onnx").write_bytes(b"not a model")
    (v / "t1.onnx.json").write_text("{}")
    with pytest.raises(ValueError, match="vacío"):
        tts.speak("  ", model=str(v))


def test_piper_text_uses_symbols_in_piper_map():
    import unicodedata

    out = tts.piper_text("¡Mba'éichapa “Ñande” róga!")
    assert out == unicodedata.normalize("NFD", "mba'eˈichapa ñande roˈga!").replace("́", "")
    assert "́" not in out and out == out.lower()


def test_speak_text_voice_gets_piper_text(tmp_path, monkeypatch):
    import numpy as np

    v = tmp_path / "t2"
    v.mkdir()
    (v / "t2.onnx").write_bytes(b"x")
    (v / "t2.onnx.json").write_text("{}")
    seen = []

    class Chunk:
        audio_float_array = np.zeros(10, np.float32)

    class Voice:
        config = type("C", (), {"sample_rate": 22050, "phoneme_type": type("P", (), {"value": "text"})()})()

        def synthesize(self, text):
            seen.append(text)
            return [Chunk()]

    monkeypatch.setattr(tts, "_load_piper", lambda model, use_cuda: Voice())
    tts.speak("Róga", model=str(v))
    assert seen == [tts.piper_text("Róga")]


def _text_voice(tmp_path, monkeypatch):
    import numpy as np

    v = tmp_path / "t2"
    v.mkdir()
    (v / "t2.onnx").write_bytes(b"x")
    (v / "t2.onnx.json").write_text("{}")
    calls = []

    class Chunk:
        audio_float_array = np.ones(10, np.float32)

    class Voice:
        config = type("C", (), {"sample_rate": 100, "phoneme_type": type("P", (), {"value": "text"})()})()

        def synthesize(self, text):
            calls.append(text)
            return [Chunk()]

    monkeypatch.setattr(tts, "_load_piper", lambda model, use_cuda: Voice())
    return str(v), calls


def test_missing_local_voice_explains_how_to_get_it(tmp_path):
    with pytest.raises(ValueError, match="tts-pick"):
        tts.speak("che", model=str(tmp_path / "nope"))
    with pytest.raises(ValueError, match="tts-pick"):
        tts.speak("che", model=str(tmp_path))  # directorio sin .onnx


def test_text_voice_rejects_digits(tmp_path, monkeypatch):
    voice, calls = _text_voice(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cifras"):
        tts.speak("Ary 2024", model=voice)
    assert calls == []


def test_text_voice_text_empty_after_mapping_raises(tmp_path, monkeypatch):
    voice, _ = _text_voice(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="vacío"):
        tts.speak("“”¡¿", model=voice)


def test_text_voice_synthesizes_sentence_by_sentence(tmp_path, monkeypatch):
    voice, calls = _text_voice(tmp_path, monkeypatch)
    audio, sr = tts.speak("Che róga. Nde róga? Ha'e róga!", model=voice)
    assert calls == [tts.piper_text("Che róga."), tts.piper_text("Nde róga?"), tts.piper_text("Ha'e róga!")]
    assert len(audio) == 3 * 10 + 2 * int(0.2 * sr)  # pausa breve entre frases
