import pytest

pytestmark = pytest.mark.slow


def test_translate_both_directions():
    from guanani.nmt import translate

    es = translate(["Mba'éichapa reiko?"], "gn", "es")
    gn = translate(["Buenos días, ¿cómo estás?"], "es", "gn")
    assert len(es) == 1 and es[0].strip()
    assert len(gn) == 1 and gn[0].strip()


def test_tts_then_stt_roundtrip_nonempty():
    from guanani.stt import SR, transcribe
    from guanani.tts import speak

    audio, sr = speak("Mba'éichapa, che réra Juan.")
    assert sr == SR and audio.ndim == 1 and len(audio) > sr // 2
    text = transcribe([audio])[0]
    assert text.strip()


def test_lora_tiny_training_merges_into_translatable_model(tmp_path):
    from guanani.nmt import translate
    from guanani.train.corpus import build_examples
    from guanani.train.lora import merge, train

    pairs = [{"gn": f"Che réra Juan {i}.", "es": f"Me llamo Juan {i}.", "source": "t"} for i in range(50)]
    dev = [{"gn": "Mba'éichapa reiko?", "es": "¿Cómo estás?"}] * 4
    base = "facebook/nllb-200-distilled-600M"
    res = train("tiny", build_examples(pairs), dev, base=base, batch_size=4, grad_accum=1,
                max_steps=20, eval_steps=10, models_dir=tmp_path)
    assert res["best_step"] in (0, 10, 20)  # el paso 0 (base) también compite
    merge(base, tmp_path / "tiny" / "best_adapter", tmp_path / "tiny")
    out = translate(["Mba'éichapa reiko?"], "gn", "es", model=str(tmp_path / "tiny"))
    assert out[0].strip()


def test_full_finetune_tiny_t5_saves_complete_model(tmp_path):
    from guanani.nmt import translate
    from guanani.train.corpus import build_examples
    from guanani.train.lora import finalize_full, train

    pairs = [{"gn": f"Che réra Juan {i}.", "es": f"Me llamo Juan {i}.", "source": "t"} for i in range(20)]
    dev = [{"gn": "Mba'éichapa reiko?", "es": "¿Cómo estás?"}] * 2
    base = "hf-internal-testing/tiny-random-t5"
    res = train("tinyfull", build_examples(pairs), dev, base=base, full=True, batch_size=4, grad_accum=1,
                max_steps=4, eval_steps=2, models_dir=tmp_path)
    out = tmp_path / "tinyfull"
    assert res["best_step"] in (0, 2, 4) and (out / "best_model").exists()
    finalize_full(out, base)
    assert not (out / "best_model").exists() and not (out / "adapter_config.json").exists()
    assert isinstance(translate(["Mba'éichapa"], "gn", "es", model=str(out))[0], str)


@pytest.mark.slow
def test_whisper_tiny_finetune_saves_best_and_transcribes(tmp_path, monkeypatch):
    from guanani import stt
    from guanani.train import stt_data, whisper

    rows, _ = stt_data.load_stt(["dev"])
    monkeypatch.setattr(stt_data, "load_stt", lambda kinds: (rows[:8], {}))
    meta = whisper.run("tiny", base="openai/whisper-tiny", max_steps=4, eval_steps=2, batch_size=4, grad_accum=1,
                       lr=1e-4, models_dir=tmp_path, dev_limit=4)
    assert set(meta["history"]) == {"0", "2", "4"}
    best = tmp_path / "tiny" / "best_model"
    assert stt.backend_of(str(best)) == "whisper"
    assert isinstance(stt.transcribe([stt.load_audio(rows[0]["audio"])], model=str(best))[0], str)


@pytest.mark.slow
def test_omnilingual_ctc_transcribes_and_local_card_works(tmp_path):
    from guanani import stt

    if not (stt.OMNI_ENV / ".venv" / "bin" / "python").exists():
        pytest.skip("sin entorno envs/omnilingual")
    audio = stt.load_audio(str(sorted((stt.Path("data/common_voice/clips")).glob("*.mp3"))[0]))
    base = stt.transcribe([audio], model="omniASR_CTC_300M_v2")[0]
    card = tmp_path / "alias"
    card.mkdir()
    (card / "card.yaml").write_text(
        "name: guanani_alias\nmodel_family: wav2vec2_asr\nmodel_arch: 300m_v2\n"
        "checkpoint: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt\n"
        "tokenizer_ref: omniASR_tokenizer_written_v2\n"
    )
    assert base.strip() and stt.transcribe([audio], model=str(card))[0] == base


@pytest.mark.slow
def test_piper_voice_speaks():
    from pathlib import Path

    from guanani import tts

    voice = Path("models/smoke")
    if not voice.exists():
        pytest.skip("sin voz de humo (Task 4 Step 5 en el servidor)")
    audio, sr = tts.speak("Mba'éichapa", model=str(voice))
    assert sr == 22050 and len(audio) > sr // 10
