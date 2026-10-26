"""Identificación de variedad con GlotLID: acepta guaraní paraguayo, rechaza otras variedades y lenguas."""

import math

ACCEPTED = frozenset({"gug_Latn", "grn_Latn"})
GLOTLID = ("cis-lmu/glotlid", "model_v3.bin")


def load_glotlid():
    try:
        import fasttext
    except ImportError as e:
        raise RuntimeError("Falta fasttext: ejecuta `uv sync` (dependencia fasttext-wheel)") from e
    from huggingface_hub import hf_hub_download

    return fasttext.load_model(hf_hub_download(*GLOTLID))


class GlotLID:
    def __init__(self, model, threshold: float = 0.0):
        self.model, self.threshold = model, threshold

    def top(self, sentence: str) -> tuple[str, float]:
        labels, probs = self.model.predict(sentence.replace("\n", " "), k=1)
        return labels[0].removeprefix("__label__"), float(probs[0])

    def accepted_prob(self, sentence: str) -> float:
        label, p = self.top(sentence)
        return p if label in ACCEPTED else 0.0

    def reject_label(self, sentence: str) -> str | None:
        label, p = self.top(sentence)
        if label not in ACCEPTED:
            return label
        return None if p >= self.threshold else f"{label}<umbral"


def calibrate_threshold(probs: list[float], recall: float = 0.95) -> float:
    # Mayor umbral que deja pasar al menos `recall` de las frases de referencia.
    ranked = sorted(probs, reverse=True)
    return ranked[max(0, math.ceil(recall * len(ranked)) - 1)]
