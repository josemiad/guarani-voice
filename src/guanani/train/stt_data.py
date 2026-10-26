"""Common Voice para entrenar STT: frases normalizadas, sin frases del test ni hablantes de test/dev."""

from guanani import data
from guanani.normalize import normalize
from guanani.train.corpus import decont_key

SPLITS = ("train", "dev", "other")


def load_stt(kinds: list[str]) -> tuple[list[dict], dict]:
    for k in kinds:
        if k not in SPLITS:
            raise ValueError(f"split de Common Voice no válido para entrenar: {k}; opciones: {list(SPLITS)}")
    test = data.load_common_voice("test")
    test_keys = {decont_key(r["gn"]) for r in test}
    # Common Voice separa train/dev/test por hablante, pero `other` no: sin este filtro
    # el 74 % de los clips de test eran de hablantes vistos al entrenar.
    test_spk = {r["speaker"] for r in test} - {""}
    dev_spk = {r["speaker"] for r in data.load_common_voice("dev")} - {""} if kinds != ["dev"] else set()
    rows, stats = [], {}
    for k in kinds:
        s = dict.fromkeys(("in", "vacia", "contaminada", "hablante_eval", "aceptada"), 0)
        for r in data.load_common_voice(k):
            s["in"] += 1
            text = normalize(r["gn"])
            if not text:
                s["vacia"] += 1
            elif r["speaker"] in test_spk or (k != "dev" and r["speaker"] in dev_spk):
                s["hablante_eval"] += 1
            elif decont_key(text) in test_keys:
                s["contaminada"] += 1
            else:
                s["aceptada"] += 1
                rows.append({"audio": r["audio"], "text": text, "split": k})
        stats[k] = s
    return rows, stats
