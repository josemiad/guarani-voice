"""Normalización de texto guaraní: NFC, puso canónico y grafías nasales heredadas."""

import re
import unicodedata

PUSO = "'"

# ponytail: mapa fijo de grafías vistas en Jojajovai; ampliar si otro corpus trae variantes nuevas.
# 'ü' no se toca: aparece en préstamos del español (itaugüeño).
_TABLE = str.maketrans(
    {c: PUSO for c in "’‘´`ʼꞌ′‚"}
    | dict(zip("âêîôûŷÿäëïö", "ãẽĩõũỹỹãẽĩõ"))
    | dict(zip("ÂÊÎÔÛŶŸÄËÏÖ", "ÃẼĨÕŨỸỸÃẼĨÕ"))
    | {c: "g̃" for c in "ĝḡğġ"}
    | {c: "G̃" for c in "ĜḠĞĠ"}
    | {"​": None, "﻿": None}
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_TABLE)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
