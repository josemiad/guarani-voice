import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from guanani import data
from guanani.train import lora, sources


def test_tokenize_examples_sets_language_codes():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    ex = [
        {"src": "che", "tgt": "yo", "src_lang": "grn_Latn", "tgt_lang": "spa_Latn"},
        {"src": "yo", "tgt": "che", "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
    ]
    out = lora.tokenize_examples(tok, ex)
    firsts = {(tok.convert_ids_to_tokens(o["input_ids"][0]), tok.convert_ids_to_tokens(o["labels"][0])) for o in out}
    assert firsts == {("grn_Latn", "spa_Latn"), ("spa_Latn", "grn_Latn")}


def test_dev_history_survives_resume(tmp_path):
    (tmp_path / "dev_history.json").write_text(
        json.dumps({"1000": {"gn-es": 40.0, "es-gn": 34.0, "mean": 37.0}, "2000": {"gn-es": 41.0, "es-gn": 35.0, "mean": 38.0}}),
        encoding="utf-8",
    )
    cb = lora.DevChrF(tok=None, dev_pairs=[], eval_steps=1000, out_dir=tmp_path)
    assert cb.best == 38.0 and cb.best_step == 2000


def test_run_rejects_unknown_data_kind(tmp_path):
    with pytest.raises(ValueError, match="foo"):
        lora.run("x", ["parallel", "foo"], models_dir=tmp_path)


def test_run_missing_bt_raises_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "parallel.jsonl").write_text('{"gn": "a", "es": "b", "source": "s"}\n', encoding="utf-8")
    with pytest.raises(data.DataMissing, match="backtranslate"):
        lora.run("x", ["parallel", "bt"], models_dir=tmp_path)


def test_fresh_run_discards_previous_history_and_adapter(tmp_path):
    (tmp_path / "dev_history.json").write_text('{"1000": {"gn-es": 50, "es-gn": 50, "mean": 50}}', encoding="utf-8")
    (tmp_path / "best_adapter").mkdir()
    (tmp_path / "checkpoints").mkdir()
    lora.prepare_out_dir(tmp_path, resume=False)
    assert not any(tmp_path.iterdir())


def test_resume_keeps_history_and_adapter(tmp_path):
    (tmp_path / "dev_history.json").write_text("{}", encoding="utf-8")
    (tmp_path / "best_adapter").mkdir()
    lora.prepare_out_dir(tmp_path, resume=True)
    assert (tmp_path / "dev_history.json").exists() and (tmp_path / "best_adapter").exists()


def test_load_examples_upsamples_and_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "parallel.jsonl", [{"gn": "che", "es": "yo", "source": "s"}])
    sources.write_jsonl(tmp_path / "train" / "bt2" / "bt2.jsonl", [{"gn": "ore", "es": "nosotros", "source": "bt:x"}])
    ex, counts = lora.load_examples(["parallel", "bt2"], upsample_parallel=3, bt_tag=True)
    assert counts == {"n_parallel": 3, "n_bt": 1, "n_examples": 7}
    assert ex[0]["src"] == "che" and ex[-1]["src"] == "[BT] nosotros"
    assert sum(e["src"].startswith("[BT]") for e in ex) == 1


def test_data_file_resolves_named_bt_and_rejects_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    assert lora.data_file("bt2")[0] == tmp_path / "train" / "bt2" / "bt2.jsonl"
    assert lora.data_file("bt")[0] == tmp_path / "train" / "bt.jsonl"
    with pytest.raises(ValueError, match="btN"):
        lora.data_file("foo")


class _RecTok:
    def __init__(self):
        self.calls = []

    def __call__(self, srcs, text_target, max_length, truncation):
        self.calls += srcs
        n = len(srcs)
        return {"input_ids": [[1]] * n, "attention_mask": [[1]] * n, "labels": [[2]] * n}


def test_tokenize_examples_madlad_prefixes_target_language():
    tok = _RecTok()
    ex = [
        {"src": "che", "tgt": "yo", "src_lang": "grn_Latn", "tgt_lang": "spa_Latn"},
        {"src": "yo", "tgt": "che", "src_lang": "spa_Latn", "tgt_lang": "grn_Latn"},
    ]
    out = lora.tokenize_examples(tok, ex, family="madlad")
    assert sorted(tok.calls) == ["<2es> che", "<2gn> yo"]
    assert len(out) == 2


def test_finalize_full_moves_best_model_and_saves_tokenizer(tmp_path, monkeypatch):
    best = tmp_path / "best_model"
    best.mkdir()
    (best / "config.json").write_text("{}", encoding="utf-8")
    (best / "model.safetensors").write_text("w", encoding="utf-8")

    class _FakeTok:
        def save_pretrained(self, d):
            (Path(d) / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(lora.AutoTokenizer, "from_pretrained", lambda base: _FakeTok())
    lora.finalize_full(tmp_path, "base")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.json", "model.safetensors", "tokenizer.json"]


def test_fresh_run_also_discards_best_model(tmp_path):
    (tmp_path / "best_model").mkdir()
    lora.prepare_out_dir(tmp_path, resume=False)
    assert not (tmp_path / "best_model").exists()


def test_load_examples_filters_by_allowed_sources_before_upsampling(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "parallel.jsonl",
                        [{"gn": "che", "es": "yo", "source": "jojajovai"}, {"gn": "nde", "es": "tú", "source": "americasnlp"}])
    sources.write_jsonl(tmp_path / "train" / "bt3" / "bt3.jsonl",
                        [{"gn": "ore", "es": "nosotros", "source": "bt:wikipedia"}, {"gn": "pe", "es": "vosotros", "source": "bt:x"}])
    (tmp_path / "train" / "bt3" / "meta.json").write_text('{"model": "google/madlad400-3b-mt"}', encoding="utf-8")
    ex, counts = lora.load_examples(["parallel", "bt3"], upsample_parallel=2, allow={"jojajovai", "bt:wikipedia"})
    assert counts == {"n_parallel": 2, "n_bt": 1, "n_examples": 5}
    assert {e["src"] for e in ex} == {"che", "yo", "nosotros"}


def test_load_examples_empty_after_filter_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "parallel.jsonl", [{"gn": "che", "es": "yo", "source": "jojajovai"}])
    sources.write_jsonl(tmp_path / "train" / "bt3" / "bt3.jsonl", [{"gn": "ore", "es": "nosotros", "source": "bt:x"}])
    (tmp_path / "train" / "bt3" / "meta.json").write_text('{"model": "google/madlad400-3b-mt"}', encoding="utf-8")
    with pytest.raises(ValueError, match="aprobada"):
        lora.load_examples(["parallel", "bt3"], allow={"jojajovai"})


def _bt_dir(tmp_path, kind, model):
    sources.write_jsonl(tmp_path / "train" / kind / f"{kind}.jsonl", [{"gn": "ore", "es": "nosotros", "source": "bt:wikipedia"}])
    if model:
        (tmp_path / "train" / kind / "meta.json").write_text(json.dumps({"model": model}), encoding="utf-8")


def test_license_gate_rejects_backtranslation_made_with_non_commercial_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "parallel.jsonl", [{"gn": "che", "es": "yo", "source": "jojajovai"}])
    _bt_dir(tmp_path, "bt2", "models/e2")
    _bt_dir(tmp_path, "bt3", "google/madlad400-3b-mt")
    allow = {"jojajovai", "bt:wikipedia"}
    with pytest.raises(ValueError, match="bt2"):
        lora.load_examples(["parallel", "bt2"], allow=allow)
    ex, counts = lora.load_examples(["parallel", "bt3"], allow=allow)
    assert counts["n_bt"] == 1


def test_license_gate_rejects_legacy_bt_without_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "parallel.jsonl", [{"gn": "che", "es": "yo", "source": "jojajovai"}])
    sources.write_jsonl(tmp_path / "train" / "bt.jsonl", [{"gn": "ore", "es": "nosotros", "source": "bt:wikipedia"}])
    with pytest.raises(ValueError, match="bt"):
        lora.load_examples(["parallel", "bt"], allow={"jojajovai", "bt:wikipedia"})


class _StepModel:
    step = 0

    def save_pretrained(self, p):
        Path(p).mkdir(parents=True, exist_ok=True)
        (Path(p) / "step").write_text(str(self.step), encoding="utf-8")


def test_per_direction_keeps_best_of_each_direction_including_step0(tmp_path, monkeypatch):
    table = {0: (40.0, 34.0), 10: (41.0, 33.0), 20: (40.5, 33.5)}
    monkeypatch.setattr(lora, "dev_chrf", lambda model, tok, dev, d: table[model.step][0 if d == "gn-es" else 1])
    m = _StepModel()
    cb = lora.DevChrF(None, [], 10, tmp_path, per_direction=True)
    cb.on_train_begin(None, SimpleNamespace(global_step=0), None, model=m)
    for step in (10, 20):
        m.step = step
        cb.on_step_end(None, SimpleNamespace(global_step=step), None, model=m)
    assert (tmp_path / "best_adapter_gn-es" / "step").read_text() == "10"
    assert (tmp_path / "best_adapter_es-gn" / "step").read_text() == "0"
    assert not (tmp_path / "best_adapter").exists()


def test_per_direction_best_survives_resume_and_step0_not_repeated(tmp_path, monkeypatch):
    (tmp_path / "dev_history.json").write_text(json.dumps({
        "0": {"gn-es": 40.0, "es-gn": 34.0, "mean": 37.0},
        "10": {"gn-es": 41.0, "es-gn": 33.0, "mean": 37.0},
    }), encoding="utf-8")

    def boom(*a):
        raise AssertionError("no debe evaluar de nuevo el paso 0")

    monkeypatch.setattr(lora, "dev_chrf", boom)
    cb = lora.DevChrF(None, [], 10, tmp_path, per_direction=True)
    assert cb.dir_best_step == {"gn-es": 10, "es-gn": 0}
    cb.on_train_begin(None, SimpleNamespace(global_step=10), None, model=_StepModel())


def test_run_per_direction_merges_one_model_per_direction(tmp_path, monkeypatch):
    monkeypatch.setattr(lora, "load_examples", lambda *a, **k: ([], {"n_examples": 0}))
    monkeypatch.setattr(lora.data, "load_flores", lambda split: [])
    monkeypatch.setattr(lora, "train", lambda name, ex, dev, models_dir, **kw: {"best_step": 0})
    calls = []
    monkeypatch.setattr(lora, "merge", lambda base, adapter, out: calls.append((adapter.name, out.name)))
    out = lora.run("e7", ["parallel"], models_dir=tmp_path, base="b", per_direction=True)
    assert sorted(calls) == [("best_adapter_es-gn", "es-gn"), ("best_adapter_gn-es", "gn-es")]
    assert (out / "train_meta.json").exists()


def test_per_direction_with_full_rejected(tmp_path):
    with pytest.raises(ValueError, match="per-direction"):
        lora.run("x", ["parallel"], models_dir=tmp_path, full=True, per_direction=True)


def test_lora_targets_depend_on_model_family():
    assert lora.lora_targets("nllb") == ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    assert lora.lora_targets("madlad") == ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
