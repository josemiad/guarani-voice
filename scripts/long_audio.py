#!/usr/bin/env python3
"""Control de audios largos: concatena clips de cv test en audios de ~90 s y mide WER con transcribe_long."""

import sys

import numpy as np

from guanani import data
from guanani.eval import load_clips, score_asr
from guanani.stt import SR, transcribe_long

model, n_audios = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
audios, refs, _ = load_clips(data.load_common_voice("test"))
gap = np.zeros(SR // 2, np.float32)
hyps, joined, i = [], [], 0
for _ in range(n_audios):
    parts, texts = [], []
    while i < len(audios) and sum(len(p) for p in parts) < 90 * SR:
        parts += [audios[i], gap]
        texts.append(refs[i])
        i += 1
    hyps.append(transcribe_long(np.concatenate(parts), model=model))
    joined.append(" ".join(texts))
print(model, score_asr(hyps, joined))
