import json
import os
import sys

import numpy as np
import pytest

from guanani import stt


def test_backend_of_hub_names():
    assert stt.backend_of("facebook/mms-1b-all") == "mms"
    assert stt.backend_of("openai/whisper-large-v3-turbo") == "whisper"
    assert stt.backend_of("omniASR_CTC_1B_v2") == "omnilingual"


def test_backend_of_local_dirs(tmp_path):
    w = tmp_path / "w1"
    w.mkdir()
    (w / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    o = tmp_path / "s1"
    o.mkdir()
    (o / "card.yaml").write_text("name: guanani_s1\n")
    m = tmp_path / "m"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"model_type": "wav2vec2"}))
    assert [stt.backend_of(str(p)) for p in (w, o, m)] == ["whisper", "omnilingual", "mms"]


def _tone(seconds):
    t = np.arange(int(seconds * stt.SR)) / stt.SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_split_long_cuts_on_silence():
    gap = np.zeros(stt.SR, dtype=np.float32)
    audio = np.concatenate([_tone(20), gap, _tone(20), gap, _tone(5)])
    chunks = stt.split_long(audio, max_s=30)
    assert len(chunks) == 2
    assert all(len(c) <= 30 * stt.SR for c in chunks)
    assert sum(len(c) for c in chunks) >= 44 * stt.SR


def test_split_long_hard_cuts_audio_without_silence():
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, 75 * stt.SR).astype(np.float32)
    chunks = stt.split_long(noise, max_s=30)
    assert len(chunks) == 3 and max(len(c) for c in chunks) <= 30 * stt.SR


def test_transcribe_long_joins_chunk_texts(monkeypatch):
    seen = []
    monkeypatch.setattr(stt, "transcribe", lambda audios, model, device: seen.append(len(audios)) or ["a", " ", "b"])
    gap = np.zeros(stt.SR, dtype=np.float32)
    assert stt.transcribe_long(np.concatenate([_tone(8), gap, _tone(8), gap, _tone(8)])) == "a b"
    assert seen == [3]



def _fake_env(tmp_path):
    env = tmp_path / "omni"
    (env / ".venv" / "bin").mkdir(parents=True)
    py = env / ".venv" / "bin" / "python"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(0o755)
    (env / "transcribe.py").write_text(
        "import argparse, json, numpy as np\n"
        "p = argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--asset-dir')\n"
        "p.add_argument('--lang'); p.add_argument('--batch-size'); p.add_argument('inp'); p.add_argument('out')\n"
        "a = p.parse_args(); n = len(np.load(a.inp).files)\n"
        "open(a.out, 'w').write(''.join(json.dumps({'text': f'{a.model}|{a.asset_dir}|{a.lang}|{i}'}) + '\\n' for i in range(n)))\n"
    )
    return env


def test_omnilingual_runs_script_in_its_env(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "OMNI_ENV", _fake_env(tmp_path))
    out = stt.transcribe([np.zeros(10, np.float32)] * 2, model="omniASR_LLM_1B_v2")
    assert out == ["omniASR_LLM_1B_v2|None|grn_Latn|0", "omniASR_LLM_1B_v2|None|grn_Latn|1"]


def test_omnilingual_ctc_gets_no_lang_and_local_card(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "OMNI_ENV", _fake_env(tmp_path))
    model = tmp_path / "s1"
    model.mkdir()
    (model / "card.yaml").write_text("name: guanani_s1\n")
    assert stt.transcribe([np.zeros(10, np.float32)], model=str(model)) == [f"guanani_s1|{model}|None|0"]


def test_omnilingual_missing_env_explains_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "OMNI_ENV", tmp_path / "nope")
    with pytest.raises(RuntimeError, match="uv sync --project envs/omnilingual"):
        stt.transcribe([np.zeros(10, np.float32)], model="omniASR_CTC_300M_v2")


def test_split_long_default_chunks_are_short():
    # Los modelos afinados con frases de Common Voice (~4 s) empeoran mucho con fragmentos de 30 s.
    gap = np.zeros(stt.SR // 2, dtype=np.float32)
    audio = np.concatenate([np.concatenate([_tone(4), gap]) for _ in range(6)])
    assert max(len(c) for c in stt.split_long(audio)) <= 10 * stt.SR


def test_backend_of_whisper_run_dir_resolves_best_model(tmp_path):
    # guanani train stt deja los pesos en models/<name>/best_model; models/<name> debe bastar.
    best = tmp_path / "w2" / "best_model"
    best.mkdir(parents=True)
    (best / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    assert stt.backend_of(str(tmp_path / "w2")) == "whisper"
    assert stt.resolve_model(str(tmp_path / "w2")) == str(best)
    assert stt.resolve_model("openai/whisper-tiny") == "openai/whisper-tiny"


def test_omnilingual_reads_card_name_from_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(stt, "OMNI_ENV", _fake_env(tmp_path))
    model = tmp_path / "prod"  # copiado/renombrado desde models/s2b
    model.mkdir()
    (model / "card.yaml").write_text("name: guanani_s2b\nmodel_family: wav2vec2_asr\n")
    assert stt.transcribe([np.zeros(10, np.float32)], model=str(model)) == [f"guanani_s2b|{model}|None|0"]


def test_omnilingual_resolves_relative_checkpoint_in_card(tmp_path, monkeypatch):
    env = _fake_env(tmp_path)
    (env / "transcribe.py").write_text(
        "import argparse, json, pathlib, re\n"
        "p = argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--asset-dir')\n"
        "p.add_argument('--lang'); p.add_argument('--batch-size'); p.add_argument('inp'); p.add_argument('out')\n"
        "a = p.parse_args(); card = (pathlib.Path(a.asset_dir) / 'card.yaml').read_text()\n"
        "open(a.out, 'w').write(json.dumps({'text': re.search(r'^checkpoint: (.*)$', card, re.M).group(1)}) + '\\n')\n"
    )
    monkeypatch.setattr(stt, "OMNI_ENV", env)
    model = tmp_path / "moved" / "s2b"
    (model / "checkpoint").mkdir(parents=True)
    (model / "card.yaml").write_text("name: guanani_s2b\ncheckpoint: checkpoint\n")
    assert stt.transcribe([np.zeros(10, np.float32)], model=str(model)) == [str((model / "checkpoint").resolve())]
