import numpy as np
import pytest
import soundfile as sf

from guanani.stt import SR, load_audio


def test_load_audio_resamples_to_16k(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(22050, dtype=np.float32), 22050)
    audio = load_audio(str(p))
    assert abs(len(audio) - SR) <= 1


def test_load_audio_empty_raises(tmp_path):
    p = tmp_path / "empty.wav"
    sf.write(p, np.zeros(0, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="vacío"):
        load_audio(str(p))


def test_speak_rejects_empty_text():
    from guanani.tts import speak

    with pytest.raises(ValueError):
        speak("   ")
