"""Transcribe con Omnilingual ASR. Se ejecuta en envs/omnilingual/.venv (fairseq2), no en el entorno principal."""

import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="tarjeta fairseq2, p. ej. omniASR_CTC_1B_v2 o guanani_s1")
p.add_argument("--asset-dir", help="directorio con card.yaml de un modelo afinado")
p.add_argument("--lang", help="código de idioma para modelos LLM, p. ej. grn_Latn")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("inp")
p.add_argument("out")
a = p.parse_args()
if a.asset_dir:
    os.environ["FAIRSEQ2_USER_ASSET_DIR"] = a.asset_dir  # antes de importar fairseq2

import torch  # noqa: E402
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: E402

audios = [{"waveform": torch.from_numpy(x), "sample_rate": 16000} for x in np.load(a.inp).values()]
pipe = ASRInferencePipeline(model_card=a.model)
kw = {"lang": [a.lang] * len(audios)} if a.lang else {}
texts = pipe.transcribe(audios, batch_size=a.batch_size, **kw)
with open(a.out, "w", encoding="utf-8") as f:
    f.writelines(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts)
