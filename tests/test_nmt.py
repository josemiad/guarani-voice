from types import SimpleNamespace

import torch

from guanani import nmt


class _Enc(dict):
    def to(self, device):
        return self


class _Tok:
    src_lang = None

    def __call__(self, texts, **kw):
        return _Enc(texts=texts)

    def convert_tokens_to_ids(self, token):
        return 0

    def batch_decode(self, gen, **kw):
        return [t.upper() for t in gen]


class _OomOnBigBatch:
    def generate(self, texts, **kw):
        if len(texts) > 2:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return texts


def test_oom_splits_batch_and_keeps_order(monkeypatch):
    monkeypatch.setattr(nmt, "_load", lambda model, device: (_Tok(), _OomOnBigBatch()))
    assert nmt.translate(["a", "b", "c", "d", "e"], "es", "es", device="cpu") == ["A", "B", "C", "D", "E"]


class _RecordsExcInfo:
    def __init__(self):
        self.exc_during_retry = []

    def generate(self, texts, **kw):
        import sys

        if len(texts) > 2:
            raise torch.OutOfMemoryError("CUDA out of memory")
        # El reintento debe ocurrir fuera del except: si no, el traceback retiene tensores de la GPU.
        self.exc_during_retry.append(sys.exc_info()[0])
        return texts


def test_oom_retry_happens_outside_except(monkeypatch):
    m = _RecordsExcInfo()
    monkeypatch.setattr(nmt, "_load", lambda model, device: (_Tok(), m))
    nmt.translate(["a", "b", "c", "d"], "es", "es", device="cpu")
    assert m.exc_during_retry and all(e is None for e in m.exc_during_retry)


class _RecordsKwargs:
    def __init__(self):
        self.kwargs = []

    def generate(self, texts, **kw):
        self.kwargs.append(kw)
        return texts


def test_sample_uses_topk_sampling_and_default_uses_beam(monkeypatch):
    m = _RecordsKwargs()
    monkeypatch.setattr(nmt, "_load", lambda model, device: (_Tok(), m))
    nmt.translate(["a"], "es", "es", device="cpu", sample=True)
    nmt.translate(["a"], "es", "es", device="cpu")
    assert (m.kwargs[0]["do_sample"], m.kwargs[0]["top_k"], m.kwargs[0]["num_beams"]) == (True, 10, 1)
    assert m.kwargs[1]["num_beams"] == 4 and "do_sample" not in m.kwargs[1]


class _T5Model:
    config = SimpleNamespace(model_type="t5")

    def __init__(self):
        self.kwargs = []

    def generate(self, texts, **kw):
        self.kwargs.append(kw)
        return texts


def test_family_of_detects_t5_as_madlad():
    assert nmt.family_of(SimpleNamespace(model_type="t5")) == "madlad"
    assert nmt.family_of(SimpleNamespace(model_type="m2m_100")) == "nllb"
    assert nmt.family_of(None) == "nllb"


def test_madlad_uses_target_prefix_and_no_forced_bos(monkeypatch):
    m = _T5Model()
    monkeypatch.setattr(nmt, "_load", lambda model, device: (_Tok(), m))
    out = nmt.translate(["ko ñe'ẽ"], "gn", "es", device="cpu")
    assert out == ["<2ES> KO ÑE'Ẽ"]
    assert "forced_bos_token_id" not in m.kwargs[0]


def test_nllb_still_forces_bos(monkeypatch):
    m = _RecordsKwargs()
    monkeypatch.setattr(nmt, "_load", lambda model, device: (_Tok(), m))
    nmt.translate(["a"], "es", "gn", device="cpu")
    assert m.kwargs[0]["forced_bos_token_id"] == 0
