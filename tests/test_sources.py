import json

import pytest

from guanani import data
from guanani.train import sources


@pytest.fixture
def ddir(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    return tmp_path


def _write_eval_sets(d):
    (d / "flores").mkdir()
    for split in ("dev", "devtest"):
        for code, text in (("gug_Latn", f"Ñe'ẽ {split}"), ("spa_Latn", f"Frase {split}")):
            (d / "flores" / f"{split}.{code}.jsonl").write_text(json.dumps({"id": 1, "text": text}), encoding="utf-8")
    (d / "jojajovai").mkdir()
    (d / "jojajovai" / "jojajovai_all.csv").write_text(
        "split,source,gn,es,tokens_gn,tokens_es\n"
        "train,abc,che róga,mi casa,x,y\n"
        "train,abc,Ñe'ẽ devtest,otra cosa distinta,x,y\n"
        "test,abc,ko mba'e,esta cosa,x,y\n",
        encoding="utf-8",
    )
    (d / "jojajovai" / "jojajovai_sample_annotations.csv").write_text(
        "source,gn,es,dialect_unified\nblogs,ko mba'e,esta cosa,guarani\n", encoding="utf-8"
    )
    (d / "common_voice").mkdir()
    (d / "common_voice" / "test.tsv").write_text("path\tsentence\na.mp3\tMba'éichapa reiko\n", encoding="utf-8")


def test_prepare_writes_parallel_and_stats_without_eval_sentences(ddir):
    _write_eval_sets(ddir)
    (ddir / "americasnlp").mkdir()
    (ddir / "americasnlp" / "train.gn").write_text("che róga\nmba'éichapa reiko!\n", encoding="utf-8")
    (ddir / "americasnlp" / "train.es").write_text("mi casa\n¿cómo estás?\n", encoding="utf-8")
    stats = sources.prepare()
    rows = sources.read_jsonl(ddir / "train" / "parallel.jsonl")
    assert rows == [{"gn": "che róga", "es": "mi casa", "source": "jojajovai"}]
    assert stats["contaminated"] == 2 and stats["duplicate"] == 1
    assert stats["agustin"] == "no descargado"
    assert json.loads((ddir / "train" / "stats.json").read_text(encoding="utf-8"))["out"] == 1


def test_americasnlp_misaligned_raises(ddir):
    (ddir / "americasnlp").mkdir()
    (ddir / "americasnlp" / "train.gn").write_text("a\nb\n", encoding="utf-8")
    (ddir / "americasnlp" / "train.es").write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="alineados"):
        sources.load_americasnlp()


def test_kuatia_missing_subset_explains_how(ddir):
    with pytest.raises(data.DataMissing, match="kuatia"):
        next(sources.iter_kuatia_docs())


def test_kuatia_record_without_text_field_lists_fields(ddir):
    d = ddir / "kuatia"
    d.mkdir()
    (d / "madlad400_full.jsonl").write_text(json.dumps({"content": "x", "lang": "gn"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content"):
        next(sources.iter_kuatia_docs())


def test_agustin_unknown_columns_is_skipped_with_note(ddir):
    import pandas as pd

    (ddir / "agustin").mkdir()
    pd.DataFrame({"prompt": ["a"], "answer": ["b"]}).to_parquet(ddir / "agustin" / "train.parquet")
    rows, note = sources.load_agustin()
    assert rows == [] and "prompt" in note


def test_wikipedia_docs_from_parquet(ddir):
    import pandas as pd

    (ddir / "wikipedia").mkdir()
    pd.DataFrame({"title": ["A", "B", "C"], "text": ["Ñe'ẽ peteĩ.", "", "Ñe'ẽ mokõi."]}).to_parquet(
        ddir / "wikipedia" / "gn.parquet"
    )
    assert list(sources.iter_wikipedia_docs()) == [("wikipedia", "Ñe'ẽ peteĩ."), ("wikipedia", "Ñe'ẽ mokõi.")]


def test_wikipedia_missing_explains_how(ddir):
    with pytest.raises(data.DataMissing, match="Wikipedia"):
        next(sources.iter_wikipedia_docs())


def test_allowed_sources_reads_approved_rows(tmp_path):
    p = tmp_path / "lic.md"
    p.write_text(
        "# Licencias\n\n"
        "| source | Origen | Comercial | Notas |\n"
        "|---|---|---|---|\n"
        "| `jojajovai` | Jojajovai | sí | x |\n"
        "| `bt:madlad400`, `bt:fineweb-2` | KuatIA | Sí | y |\n"
        "| `bt:guarania-scraped` | KuatIA | desconocido | z |\n"
        "| `americasnlp` | AmericasNLP | no | w |\n",
        encoding="utf-8",
    )
    assert sources.allowed_sources(p) == {"jojajovai", "bt:madlad400", "bt:fineweb-2"}


def test_allowed_sources_without_table_raises(tmp_path):
    p = tmp_path / "lic.md"
    p.write_text("sin tabla\n", encoding="utf-8")
    with pytest.raises(ValueError, match="commercial"):
        sources.allowed_sources(p)


def test_real_license_table_parses_and_uses_known_sources():
    from pathlib import Path

    allowed = sources.allowed_sources(Path("docs/data-licenses.md"))
    assert allowed == {"jojajovai", "cv17", "bt:wikipedia", "bt:madlad400", "bt:fineweb-2", "bt:hplt-3",
                       "bt:FinePDF", "bt:culturax", "bt:guarania-scraped"}


def test_allowed_sources_reads_english_table(tmp_path):
    p = tmp_path / "lic.md"
    p.write_text(
        "| source | Origin | Commercial | Notes |\n"
        "|---|---|---|---|\n"
        "| `jojajovai` | J | yes | x |\n"
        "| `americasnlp` | A | unknown | y |\n",
        encoding="utf-8",
    )
    assert sources.allowed_sources(p) == {"jojajovai"}


def test_allowed_sources_only_reads_first_table(tmp_path):
    p = tmp_path / "lic.md"
    p.write_text(
        "| source | Origen | Comercial | Notas |\n"
        "|---|---|---|---|\n"
        "| `jojajovai` | J | sí | x |\n"
        "\n## Otra tabla\n\n"
        "| Recurso | A | B | C |\n"
        "|---|---|---|---|\n"
        "| `intruso` | x | sí | y |\n",
        encoding="utf-8",
    )
    assert sources.allowed_sources(p) == {"jojajovai"}
