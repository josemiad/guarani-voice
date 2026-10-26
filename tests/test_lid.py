from guanani.train.lid import GlotLID, calibrate_threshold


class _Fake:
    def __init__(self, table):
        self.table = table

    def predict(self, s, k=1):
        label, p = self.table[s]
        return (f"__label__{label}",), [p]


def test_reject_label_and_accepted_prob():
    lid = GlotLID(_Fake({"a": ("gug_Latn", 0.99), "b": ("gun_Latn", 1.0), "c": ("grn_Latn", 0.4), "d": ("spa_Latn", 0.9)}), 0.5)
    assert [lid.reject_label(s) for s in "abcd"] == [None, "gun_Latn", "grn_Latn<umbral", "spa_Latn"]
    assert lid.accepted_prob("a") == 0.99 and lid.accepted_prob("b") == 0.0


def test_calibrate_threshold_keeps_95_percent():
    probs = [1.0] * 90 + [0.8] * 5 + [0.3] * 4 + [0.0]
    t = calibrate_threshold(probs)
    assert t == 0.8
    assert sum(p >= t for p in probs) / len(probs) >= 0.95
