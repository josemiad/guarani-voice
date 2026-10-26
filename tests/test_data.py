import json

import pytest

from guanani import data


@pytest.fixture
def ddir(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    return tmp_path


def test_missing_dataset_explains_how(ddir):
    with pytest.raises(data.DataMissing, match="download.sh"):
        data.load_jojajovai()


def test_jojajovai_filters_split(ddir):
    d = ddir / "jojajovai"
    d.mkdir()
    (d / "jojajovai_all.csv").write_text(
        "split,source,gn,es,tokens_gn,tokens_es\n"
        'train,abc,"ka\'i","mono",x,y\n'
        'test,abc,"mba\'e, porã","algo, bueno",x,y\n',
        encoding="utf-8",
    )
    assert data.load_jojajovai("test") == [{"gn": "mba'e, porã", "es": "algo, bueno", "source": "abc"}]


def test_jojajovai_annotated_has_dialect(ddir):
    d = ddir / "jojajovai"
    d.mkdir()
    (d / "jojajovai_sample_annotations.csv").write_text(
        "source,gn,es,dialect_unified\nblogs,che,yo,jopara\n", encoding="utf-8"
    )
    assert data.load_jojajovai_annotated() == [{"gn": "che", "es": "yo", "source": "blogs", "dialect": "jopara"}]


def test_flores_aligns_by_id(ddir):
    d = ddir / "flores"
    d.mkdir()
    (d / "devtest.gug_Latn.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "text": t}) for i, t in [(1, "peteĩ"), (2, "mokõi")]), encoding="utf-8"
    )
    (d / "devtest.spa_Latn.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "text": t}) for i, t in [(2, "dos"), (1, "uno")]), encoding="utf-8"
    )
    assert data.load_flores() == [{"gn": "peteĩ", "es": "uno"}, {"gn": "mokõi", "es": "dos"}]


def test_common_voice_tsv_with_quotes(ddir):
    d = ddir / "common_voice"
    d.mkdir()
    (d / "test.tsv").write_text('client_id\tpath\tsentence\nx\ta.mp3\tHe\'i "che"\n', encoding="utf-8")
    rows = data.load_common_voice()
    assert rows == [{"audio": str(d / "clips" / "a.mp3"), "gn": 'He\'i "che"', "speaker": "x"}]
