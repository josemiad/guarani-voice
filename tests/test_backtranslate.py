import pytest

from guanani import nmt
from guanani.train import backtranslate as bt
from guanani.train import sources


@pytest.fixture
def ddir(tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    sources.write_jsonl(tmp_path / "train" / "bt2" / "source.jsonl",
                        [{"gn": f"frase {i}", "source": "madlad400"} for i in range(25)])
    (tmp_path / "train" / "bt.jsonl").write_text("sentinela\n", encoding="utf-8")
    calls = []

    def fake_translate(texts, src, tgt, model=None, batch_size=16, sample=False, **kw):
        assert (src, tgt) == ("gn", "es")
        calls.append((len(texts), sample, batch_size))
        return [t.upper() for t in texts]

    monkeypatch.setattr(nmt, "translate", fake_translate)
    return tmp_path, calls


def test_writes_named_run_without_touching_e2_file(ddir):
    tmp, calls = ddir
    out = bt.backtranslate("bt2", n=25, chunk=10, lid=False, sample=True, batch_size=64)
    assert out == tmp / "train" / "bt2" / "bt2.jsonl"
    rows = sources.read_jsonl(out)
    assert [r["gn"] for r in rows] == [f"frase {i}" for i in range(25)]
    assert rows[3] == {"gn": "frase 3", "es": "FRASE 3", "source": "bt:madlad400"}
    assert calls == [(10, True, 64), (10, True, 64), (5, True, 64)]
    assert (tmp / "train" / "bt.jsonl").read_text(encoding="utf-8") == "sentinela\n"


def test_resumes_without_duplicates(ddir):
    tmp, calls = ddir
    bt.backtranslate("bt2", n=25, chunk=10, lid=False)
    parts = tmp / "train" / "bt2" / "parts"
    (parts / "00001.jsonl").unlink()
    (parts / "00002.jsonl").unlink()
    (parts / "00002.jsonl.tmp").write_text("{\"parcial", encoding="utf-8")
    calls.clear()
    rows = sources.read_jsonl(bt.backtranslate("bt2", n=25, chunk=10, lid=False))
    assert [c[0] for c in calls] == [10, 5]
    assert [r["gn"] for r in rows] == [f"frase {i}" for i in range(25)]


def test_changed_settings_refuse_to_mix(ddir):
    bt.backtranslate("bt2", n=25, chunk=10, lid=False)
    with pytest.raises(ValueError, match="bt2"):
        bt.backtranslate("bt2", n=25, chunk=5, lid=False)


def test_select_sentences_takes_all_wikipedia_then_fills_with_kuatia(monkeypatch):
    monkeypatch.setattr(bt, "eval_keys", lambda: set())
    monkeypatch.setattr(bt, "iter_wikipedia_docs", lambda: iter([
        ("wikipedia", "Ko ñe'ẽ ipuku porã ha iporã. Ñande ñe'ẽ oñeñe'ẽ Paraguái retãme."),
    ]))
    monkeypatch.setattr(bt, "iter_kuatia_docs", lambda: iter([
        ("madlad400", f"Ko'ã mba'e heta {i} jey oiko ko'ápe.") for i in range(10)
    ] + [("madlad400", "Ko ñe'ẽ ipuku porã ha iporã.")]))
    rows, stats = bt.select_sentences(n=5, lid=False)
    assert [r["source"] for r in rows] == ["wikipedia", "wikipedia", "madlad400", "madlad400", "madlad400"]
    assert stats["n"] == 5 and stats["lid_threshold"] is None
    assert stats["counts"]["madlad400:aceptada"] == 10 and stats["counts"]["madlad400:duplicada"] == 1


def test_select_sentences_with_fewer_candidates_than_n(monkeypatch):
    monkeypatch.setattr(bt, "eval_keys", lambda: set())
    monkeypatch.setattr(bt, "iter_wikipedia_docs", lambda: iter([("wikipedia", "Ko ñe'ẽ ipuku porã ha iporã.")]))
    monkeypatch.setattr(bt, "iter_kuatia_docs", lambda: iter([]))
    rows, stats = bt.select_sentences(n=500, lid=False)
    assert len(rows) == 1 and stats["n"] == 1


@pytest.mark.parametrize("name", ["bt", "foo", "bt2x"])
def test_name_must_be_btN(name, tmp_path, monkeypatch):
    monkeypatch.setenv("GUANANI_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="btN"):
        bt.backtranslate(name, n=1, lid=False)
    assert not (tmp_path / "train" / name).exists()


def test_from_reuses_source_without_touching_origin(ddir):
    tmp, calls = ddir
    origin = tmp / "train" / "bt2" / "source.jsonl"
    before = origin.read_bytes()
    out = bt.backtranslate("bt3", from_="bt2", chunk=10, lid=False)
    assert origin.read_bytes() == before
    assert sorted(p.name for p in (tmp / "train" / "bt2").iterdir()) == ["source.jsonl"]
    assert [r["gn"] for r in sources.read_jsonl(out)] == [f"frase {i}" for i in range(25)]


def test_from_missing_origin_raises(ddir):
    from guanani import data

    with pytest.raises(data.DataMissing):
        bt.backtranslate("bt4", from_="bt9", lid=False)
