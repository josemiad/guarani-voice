import numpy as np
import pytest
import soundfile as sf

from guanani.eval import load_clips, run, write_report


def _report(**kw):
    base = {
        "date": "2026-09-23T1200",
        "component": "nmt",
        "model": "facebook/nllb-200-distilled-600M",
        "dataset": "flores",
        "direction": "gn-es",
        "n": 3,
        "device": "cpu",
        "metrics": {"chrf++": 40.1, "bleu": 9.5, "por_dialecto": {"jopara": {"chrf++": 30.0}}},
        "latency_s_per_item": 0.2,
    }
    return base | kw


def test_report_filename_has_no_slash_and_lives_in_results(tmp_path):
    path = write_report(_report(), tmp_path)
    assert path.parent == tmp_path
    assert "/" not in path.name and path.name.endswith(".json")
    assert "facebook-nllb-200-distilled-600M" in path.name


def test_summary_header_once_rows_appended(tmp_path):
    write_report(_report(), tmp_path)
    write_report(_report(date="2026-09-23T1300", rtf=0.05, latency_s_per_item=None), tmp_path)
    text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert text.count("| date |") == 1
    assert text.count("| nmt |") == 2
    assert "chrf++=40.1 bleu=9.5" in text
    assert "por_dialecto" not in text
    assert "RTF 0.05" in text


def test_load_clips_skips_broken_audio(tmp_path):
    good = tmp_path / "good.wav"
    sf.write(good, np.zeros(16000, dtype=np.float32), 16000)
    empty = tmp_path / "empty.wav"
    sf.write(empty, np.zeros(0, dtype=np.float32), 16000)
    corrupt = tmp_path / "corrupt.mp3"
    corrupt.write_bytes(b"no es audio")
    rows = [{"audio": str(good), "gn": "a"}, {"audio": str(empty), "gn": "b"}, {"audio": str(corrupt), "gn": "c"}]
    audios, refs, skipped = load_clips(rows)
    assert len(audios) == 1 and refs == ["a"] and skipped == 2


def test_run_rejects_incompatible_dataset(tmp_path):
    with pytest.raises(ValueError, match="stt"):
        run("stt", "flores", results_dir=str(tmp_path))


def test_nmt_eval_frees_translation_model_before_comet(monkeypatch):
    from functools import cache

    from guanani import eval as ev
    from guanani import nmt

    @cache
    def fake_load(model, device):
        return object()

    monkeypatch.setattr(nmt, "_load", fake_load)
    monkeypatch.setattr(nmt, "translate", lambda srcs, *a, **k: (nmt._load("m", "cpu"), srcs)[1])
    seen = {}

    def fake_comet(srcs, hyps, refs):
        seen["cached_models"] = nmt._load.cache_info().currsize
        return 0.5

    monkeypatch.setattr(ev, "score_comet", fake_comet)
    ev._eval_nmt([{"gn": "a", "es": "b"}], "m", "gn-es", "cpu", comet=True)
    assert seen["cached_models"] == 0
