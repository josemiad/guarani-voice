"""Traducción gn↔es con NLLB o MADLAD-400."""

from functools import cache

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from guanani.device import pick_device
from guanani.normalize import normalize

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
CODES = {"gn": "grn_Latn", "es": "spa_Latn"}
MADLAD_PREFIX = {"gn": "<2gn> ", "es": "<2es> "}


def family_of(config) -> str:
    return "madlad" if getattr(config, "model_type", None) == "t5" else "nllb"


def load_seq2seq(model: str, dtype: torch.dtype):
    # Safetensors primero: evita volver a descargar pytorch_model.bin si solo está en caché la conversión.
    # Si no hay safetensors, o la conversión omite pesos compartidos (NLLB-600M), se usa el .bin.
    try:
        m = AutoModelForSeq2SeqLM.from_pretrained(model, dtype=dtype, use_safetensors=True)
    except OSError:
        return AutoModelForSeq2SeqLM.from_pretrained(model, dtype=dtype, use_safetensors=False)
    if any(p.is_meta for p in m.parameters()):
        m = AutoModelForSeq2SeqLM.from_pretrained(model, dtype=dtype, use_safetensors=False)
    return m


@cache
def _load(model: str, device: str):
    tok = AutoTokenizer.from_pretrained(model)
    # T5 (MADLAD) desborda en fp16: bf16 en GPU.
    half = torch.bfloat16 if family_of(AutoConfig.from_pretrained(model)) == "madlad" else torch.float16
    return tok, load_seq2seq(model, half if device == "cuda" else torch.float32).to(device).eval()


def translate(
    texts: list[str],
    src: str,
    tgt: str,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 16,
    sample: bool = False,
) -> list[str]:
    device = pick_device(device)
    tok, m = _load(model, device)
    if src == "gn":
        texts = [normalize(t) for t in texts]
    # Muestreo top-k para retrotraducción (mejor que beam a gran escala); beam 4 para evaluar.
    gen = {"do_sample": True, "top_k": 10, "num_beams": 1} if sample else {"num_beams": 4}
    if family_of(getattr(m, "config", None)) == "madlad":
        texts = [MADLAD_PREFIX[tgt] + t for t in texts]
    else:
        tok.src_lang = CODES[src]
        gen["forced_bos_token_id"] = tok.convert_tokens_to_ids(CODES[tgt])
    if sample:
        torch.manual_seed(0)
    out = []
    for i in range(0, len(texts), batch_size):
        out += _generate(tok, m, texts[i : i + batch_size], device, gen)
    return [normalize(o) for o in out] if tgt == "gn" else out


def _generate(tok, m, texts: list[str], device: str, gen: dict) -> list[str]:
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    oom = False
    try:
        with torch.inference_mode():
            out = m.generate(**enc, max_new_tokens=256, **gen)
    except torch.OutOfMemoryError:
        if len(texts) == 1:
            raise
        oom = True
    if not oom:
        return tok.batch_decode(out, skip_special_tokens=True)
    # Lotes con frases largas no caben en GPUs compartidas: se parte el lote en dos.
    # Fuera del except, para que el traceback no retenga los tensores del intento fallido.
    del enc
    torch.cuda.empty_cache()
    half = len(texts) // 2
    return _generate(tok, m, texts[:half], device, gen) + _generate(tok, m, texts[half:], device, gen)
