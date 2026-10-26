from guanani.eval import asr_clean, by_group, score_asr, score_mt


def test_score_mt_perfect_and_bad():
    assert score_mt(["che ha'e"], ["che ha'e"])["chrf++"] == 100.0
    assert score_mt(["xyz"], ["che ha'e"])["chrf++"] < 20


def test_score_mt_rewards_shared_morphemes():
    partial = score_mt(["ojapo hikuái"], ["ojapóta hikuái"])["chrf++"]
    assert 30 < partial < 100


def test_asr_clean_keeps_puso_and_combining_tilde():
    assert asr_clean("¡Ka’i, G̃uahẽ!") == "ka'i g̃uahẽ"


def test_score_asr_counts_deletion():
    s = score_asr(["che ha'e", "mba'e"], ["che ha'e", "mba'e porã"])
    assert s["wer"] == 0.25
    assert s["empty_refs"] == 0


def test_score_asr_skips_empty_refs_and_allows_empty_hyp():
    s = score_asr(["", "abc"], ["che", "..."])
    assert s["wer"] == 1.0
    assert s["empty_refs"] == 1


def test_by_group_splits_scores():
    out = by_group(["a b", "x"], ["a b", "c d"], ["guarani", "jopara"])
    assert out["guarani"]["chrf++"] == 100.0
    assert out["guarani"]["n"] == 1
    assert out["jopara"]["chrf++"] < 100.0


def test_eval_tts_uses_given_asr_model_and_utmos(tmp_path, monkeypatch):
    import numpy as np

    from guanani import eval as ev
    from guanani import stt, tts

    seen = {}
    monkeypatch.setattr(tts, "speak", lambda text, model, device: (np.zeros(16000, np.float32), 16000))
    monkeypatch.setattr(stt, "transcribe", lambda wavs, model, device: seen.setdefault("asr", model) and ["a"] * len(wavs))
    monkeypatch.setattr(ev, "score_utmos", lambda wavs, sr: 3.5)
    rep = ev._eval_tts([{"gn": "a"}], "voz", "cpu", tmp_path / "audio", asr_model="models/w2", utmos=True)
    assert seen["asr"] == "models/w2" and rep["asr_model"] == "models/w2"
    assert rep["metrics"]["utmos"] == 3.5 and rep["metrics"]["roundtrip_cer"] == 0.0


def test_eval_tts_counts_rejected_sentences_as_failures(tmp_path, monkeypatch):
    import numpy as np

    from guanani import eval as ev
    from guanani import stt, tts

    def fake_speak(text, model, device):
        if "2024" in text:
            raise ValueError("las cifras no están soportadas")
        return np.zeros(16000, np.float32), 16000

    monkeypatch.setattr(tts, "speak", fake_speak)
    monkeypatch.setattr(stt, "transcribe", lambda wavs, model, device: ["a"] * len(wavs))
    rep = ev._eval_tts([{"gn": "a"}, {"gn": "ary 2024"}], "voz", "cpu", tmp_path / "audio", utmos=False)
    # Misma lista de frases que la referencia: la rechazada cuenta como fallo, no se descarta.
    assert rep["n"] == 2 and rep["rejected"] == 1
    assert rep["metrics"]["roundtrip_cer"] > 0


def test_eval_tts_reports_cer_without_digit_sentences(tmp_path, monkeypatch):
    import numpy as np

    from guanani import eval as ev
    from guanani import stt, tts

    monkeypatch.setattr(tts, "speak", lambda text, model, device: (np.zeros(16000, np.float32), 16000))
    monkeypatch.setattr(stt, "transcribe", lambda wavs, model, device: ["che", "x"])
    rep = ev._eval_tts([{"gn": "che"}, {"gn": "ary 2024"}], "voz", "cpu", tmp_path / "audio", utmos=False)
    assert rep["metrics"]["roundtrip_cer_sin_cifras"] == 0.0 and rep["metrics"]["roundtrip_cer"] > 0
