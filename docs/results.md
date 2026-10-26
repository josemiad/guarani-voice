# Measured results

This page brings together every measured result for the system's three components: translation (NMT), speech recognition (STT) and speech synthesis (TTS). Each row comes from a JSON report in `results/` that `guanani eval` generated. For the reasoning behind each experiment, in chronological order, see [`results/SUMMARY.md`](../results/SUMMARY.md).

✅ means the threshold was met and ❌ means it was not. "Commercial" means a licence that allows commercial use: Apache 2.0, MIT, or our own.

## Final status

| Component | Best commercial model | Main metric | Threshold | Met? |
|---|---|---|---|---|
| NMT es→gn | MADLAD-400 3B base (`models/e7/es-gn`) | FLORES+ chrF++ 34.24 | ≥ 36.05 | ❌ (−1.8) |
| NMT gn→es | E5, fine-tuned MADLAD-3B (`models/e5`) | FLORES+ chrF++ 39.49 | ≥ 39.85 | ❌ (−0.36) |
| NMT Jopara gn→es | E5 | Annotated Jojajovai chrF++ 52.69 | ≥ 49.31 | ✅ |
| STT | W2, fine-tuned Whisper large-v3-turbo (`models/w2`) | CV17 test WER 21.3 % / CER 3.6 % | WER ≤ 41.6 % (target ≤ 35 %), CER ≤ 5.8 % | ✅ |
| TTS | T2, Piper trained from scratch on letters (`models/t2`) | Round-trip CER without digits 12.1 %; UTMOS 2.57; CPU RTF 0.035 | CER ≤ 13.9 %, UTMOS ≥ 3.18, RTF ≤ 0.2 | Speed only ✅ |

- **STT is ready.** It beats the non-commercial reference (MMS) by 20 WER points.
- **NMT falls just short.** Both directions are less than 2 chrF++ below the threshold. The remaining lever is high-quality, general-domain Guarani data.
- **TTS is fast but less intelligible and less natural** than MMS-TTS (CC-BY-NC). The voice is a prototype from one Common Voice donor and must be replaced before launch.

## Methodology

| Component | Test set | n | Metrics |
|---|---|---|---|
| NMT | FLORES+ devtest (primary reference) | 1,012 | chrF++, BLEU (sacrebleu), COMET (`wmt22-comet-da`; indicative only into gn, which COMET does not cover) |
| NMT | Jojajovai test | 5,334 | same (training domain: good for comparing variants, not as an absolute figure) |
| NMT | Annotated Jojajovai (formal + Jopara sample) | 400 | same |
| STT | Common Voice 17 gn test (`asr_clean`) | 855 | WER, CER (jiwer), RTF |
| TTS | First 200 sentences of FLORES+ devtest | 200 | Round-trip CER/WER (synthesis → W2), UTMOS, RTF |

**Thresholds.**
- NMT (subproject 1): es→gn must reach the NLLB-1.3B base + 3 chrF++ (36.05); gn→es must not fall below the base (39.85); Jopara must reach the base (49.31).
- STT (subproject 2): match MMS-1B-all (CER ≤ 5.8 %, WER ≤ 41.6 %), with a target of WER ≤ 35 %.
- TTS (subproject 3): CER and UTMOS at least as good as MMS-TTS measured with the same recogniser, and CPU RTF ≤ 0.2.

**Hardware.** The baselines (subproject 0) and E1 ran on a shared 12 GB RTX 3060. Everything else ran on a 24 GB RTX PRO 5000 Blackwell. Latencies measured on the two GPUs are not comparable, even though both appear as `cuda`.

**Caveats.**
- MMS and Omnilingual may have seen Common Voice during pre-training, so their scores without fine-tuning may be inflated.
- NLLB may have seen text from the Jojajovai domain.
- No metric has been checked with human speakers or listeners yet.

## NMT: Guarani ↔ Spanish translation

### Models

| Id | Model | Training | Licence |
|---|---|---|---|
| — | NLLB-200 distilled 600M / 1.3B | not fine-tuned | CC-BY-NC |
| E1 | NLLB-1.3B + LoRA r=16 | 22,095 real pairs (Jojajovai + AmericasNLP), 2 epochs | CC-BY-NC |
| E2 | NLLB-1.3B + LoRA | E1 + 60,000 back-translated KuatIA sentences | CC-BY-NC |
| E3 | NLLB-1.3B + LoRA | pairs ×4 + 499,420 back-translated (`bt2`, GlotLID), 1 epoch | CC-BY-NC |
| E4 | NLLB-1.3B + LoRA | E3 with a `[BT]` tag | CC-BY-NC |
| — | MADLAD-400 3B / 7B | not fine-tuned | Apache 2.0 |
| E5 | MADLAD-3B, full fine-tuning (bf16, Adafactor) | 19,902 licence-cleared pairs ×4 + 498,957 back-translated (`bt3`) | Apache 2.0 |
| E7 | MADLAD-3B + LoRA, best checkpoint per direction | same data as E5; es→gn = step 0 (base) | Apache 2.0 |

### FLORES+ devtest (n = 1,012)

| Model | Licence | es→gn chrF++ | es→gn BLEU | es→gn COMET* | gn→es chrF++ | gn→es BLEU | gn→es COMET | s/sentence GPU (es→gn / gn→es) |
|---|---|---|---|---|---|---|---|---|
| NLLB-600M | CC-BY-NC | 31.75 | 6.75 | 0.557 | 37.96 | 13.90 | 0.649 | 0.095 / 0.072 |
| NLLB-1.3B | CC-BY-NC | 33.05 | 7.29 | 0.566 | 39.85 | 15.71 | 0.689 | 0.141 / 0.105 |
| E1 | CC-BY-NC | 32.40 | 7.12 | 0.561 | 40.32 | 16.00 | 0.689 | 0.142 / 0.115 |
| E2 | CC-BY-NC | 33.54 | 6.69 | 0.566 | **40.63** | 16.16 | 0.690 | 0.077 / 0.058 |
| E3 | CC-BY-NC | 33.67 | 7.09 | 0.570 | 40.52 | 16.10 | 0.690 | 0.079 / 0.058 |
| E4 | CC-BY-NC | 33.34 | 7.23 | 0.567 | 40.58 | **16.19** | 0.690 | 0.075 / 0.058 |
| MADLAD-3B | Apache 2.0 | 34.12 | 7.02 | 0.587 | 38.13 | 14.21 | 0.687 | 0.197 / 0.269 |
| MADLAD-7B | Apache 2.0 | 29.10 | 4.88 | 0.568 | 38.17 | 14.50 | 0.694 | 0.595 / 0.602 |
| E5 | Apache 2.0 | 32.00 | 5.74 | 0.565 | 39.49 | 15.46 | **0.699** | 0.214 / 0.119 |
| E7 (per direction) | Apache 2.0 | **34.24** | 6.99 | **0.588** | 39.40 | 15.35 | 0.694 | 0.185 / 0.122 |
| Threshold | | ≥ 36.05 | | | ≥ 39.85 | | | |

\* Indicative only into Guarani.

### Jojajovai

| Model | test es→gn chrF++ / BLEU (n = 5,334) | test gn→es chrF++ / BLEU (n = 5,334) | annotated gn→es chrF++ / BLEU (n = 400) | annotated es→gn chrF++ / BLEU (n = 400) |
|---|---|---|---|---|
| NLLB-600M | 47.03 / 23.91 | 51.75 / 30.91 | 46.69 / 26.36 | 41.18 / 18.69 |
| NLLB-1.3B | 50.50 / 27.40 | 54.34 / 34.08 | 49.31 / 29.24 | 45.00 / 21.43 |
| E1 | 52.47 / 29.28 | 55.21 / 34.99 | 50.51 / 30.75 | — |
| E2 | 52.68 / 28.02 | 55.54 / 35.28 | 50.88 / 30.73 | — |
| E3 | 52.76 / 27.79 | 55.40 / 35.08 | 50.76 / 30.49 | — |
| E4 | 52.97 / 29.06 | 55.36 / 35.12 | 51.01 / 30.76 | — |
| MADLAD-3B | 38.62 / 13.71 | 47.48 / 26.91 | 42.39 / 22.78 ❌ | — |
| E5 | **53.09** / **29.56** | **56.87** / **37.77** | **52.69** / **33.30** ✅ | — |
| E7 | 38.65 / 13.65 | 56.21 / 37.13 | 51.45 / 32.35 ✅ | — |

gn→es COMET on test / annotated: NLLB-1.3B 0.726 / 0.708; E5 0.741 / 0.725; E7 0.737 / 0.715. The other COMET values are in the JSON files.

### CPU latency (VPS scenario; FLORES gn→es, n = 50, transformers fp32, beam 4)

| Model | chrF++ | BLEU | s/sentence |
|---|---|---|---|
| NLLB-600M | 38.60 | 14.78 | 2.31 |
| NLLB-1.3B | 41.34 | 17.31 | 3.99 |

Both are above the < 1 s per sentence target. Serving on CPU will need CTranslate2 INT8.

### External reference

Google Translate on 200 FLORES devtest sentences, scored in blocks: chrF++ 40.41. Scored the same way, NLLB-1.3B gets 38.57 and E1 gets 37.70.

### Takeaways

- **es→gn is the weak spot.** No fine-tuned model beats the MADLAD-3B base on FLORES. Fine-tuning learns the Jojajovai domain (+14.5 chrF++ on its test set) at the cost of generalisation (−2.1 on FLORES for E5).
- Back-translation adds little: E2 − E1 = +1.1 chrF++ es→gn, and scaling it 8× (E3) adds only +0.13 more.
- MADLAD-7B is worse than 3B for es→gn (−5 chrF++).
- The MADLAD base misses the Jopara threshold (42.39), so gn→es needs a fine-tuned model.

### Training times (RTX PRO 5000)

| Run | Time |
|---|---|
| E1 (on the RTX 3060) | 3.0 h |
| E2 | 1.5 h |
| `bt2` back-translation | 2.5 h |
| E3 | 4.7 h |
| E4 | 4.8 h |
| `bt3` back-translation | 4.2 h |
| E5 | 12 h |
| E7 | 12.6 h |

## STT: Guarani speech → text

Test set: Common Voice 17 gn test, 855 clips. Training data is `train` (1,565 clips, 1.7 h). The "+ other" models add 7,614 filtered, unvalidated clips, for 9,179 clips and 11.0 h in total. Sentences and speakers from test and dev are excluded from training.

| Id | Model | Data | Licence | WER | CER | GPU RTF |
|---|---|---|---|---|---|---|
| — | MMS-1B-all (reference) | — | CC-BY-NC | 41.6 % | 5.8 % | 0.024 |
| S0 | Omnilingual CTC-300M v2 | not fine-tuned | Apache 2.0 | 62.9 % | 13.5 % | 0.003 |
| S0 | Omnilingual CTC-1B v2 | not fine-tuned | Apache 2.0 | 50.2 % | 10.3 % | 0.014 |
| S0 | Omnilingual CTC-3B v2 | not fine-tuned | Apache 2.0 | 38.9 % | 7.5 % | 0.502 |
| S0 | Omnilingual LLM-1B v2 | not fine-tuned | Apache 2.0 | 36.4 % | 6.8 % | 0.076 |
| W1 | Whisper large-v3-turbo fine-tuned | train | MIT | 31.9 % ✅ | 5.4 % ✅ | 0.013 |
| S1 | Omnilingual CTC-1B fine-tuned | train | Apache 2.0 | 30.7 % ✅ | 4.4 % ✅ | 0.005 |
| S2b | Omnilingual CTC-1B fine-tuned | train + other | Apache 2.0 | 23.1 % ✅ | **3.2 %** ✅ | 0.005 |
| **W2** | Whisper large-v3-turbo fine-tuned | train + other | MIT | **21.3 %** ✅ | 3.6 % ✅ | 0.013 |
| Threshold | | | | ≤ 41.6 % (target ≤ 35 %) | ≤ 5.8 % | |

**CPU:** MMS-1B-all on 50 clips gets WER 49.5 %, CER 6.5 % and RTF 0.28.

**Long audio** (an indicative check with no JSON report behind it): test clips were joined into ~90 s recordings and split on silences. With 10 s chunks, W2 gets WER 30.7 % / CER 13.2 % and S2b gets 30.3 % / 7.3 %. With 30 s chunks the WER is 38 % for W2, 42 % for S2b and 50 % for MMS. On long audio, S2b is more reliable than W2.

**Corrected leak:** an earlier S2 model was trained on clips from test speakers and scored 22.9 % / 3.1 %. It was discarded. S2b, retrained without those speakers, scores almost the same.

**Training times:** W1 ~1 h 45 min; W2 ~1 h 35 min; S1 and S2b ~5.5 h each.

## TTS: Guarani text → speech

Test set: 200 FLORES+ devtest sentences. Intelligibility is the round-trip CER: synthesise the sentence, then transcribe it with W2. 59 of the sentences contain digits. T2 rejects digits and each rejected sentence counts as a failure, so the "without digits" column (141 sentences) is the fair comparison between voices.

### Final evaluation (2026-10-03)

| Id | Voice | Licence | CER (200) | CER without digits (141) | WER (200) | UTMOS | GPU RTF | CPU RTF |
|---|---|---|---|---|---|---|---|---|
| T0 | MMS-TTS-grn | CC-BY-NC | **13.9 %** | **10.9 %** | 53.7 % | **3.18** | 0.004 | 0.25 |
| T1 | Piper medium, eSpeak-NG `gn` phonemes | commercial | 17.0 % | 12.5 % | 56.4 % | 2.54 | 0.025 | 0.035 ✅ |
| **T2** | Piper medium, letters | commercial | 39.0 %¹ | 12.1 % | 67.1 % | 2.57 | 0.026 | **0.035** ✅ |
| Threshold | | | ≤ 13.9 % | | | ≥ 3.18 | | ≤ 0.2 |

¹ Includes the 59 sentences rejected for containing digits.

CPU RTF comes from 20-sentence runs. Piper always synthesises on CPU with onnxruntime; where the JSON files say `cuda`, that is the recogniser's device.

### Earlier measurements

| Date | Voice | Recogniser | n | CER | WER | UTMOS | RTF |
|---|---|---|---|---|---|---|---|
| 2026-09-23 | MMS-TTS | MMS-1B-all | 200 | 11.7 % | 52.4 % | — | 0.010 GPU |
| 2026-09-24 | MMS-TTS | MMS-1B-all | 20 | 14.1 % | 53.6 % | — | 0.19 CPU |
| 2026-10-02 | T1 | W2 | 200 | 19.0 % | 59.6 % | 2.55 | 0.027 |
| 2026-10-02 | T2 (before digits were rejected) | W2 | 200 | 14.8 % | 56.5 % | 2.55 | 0.025 |

**Piper synthesis is not deterministic.** T1 scored 19.0 % and 17.0 % CER in two runs on the same sentences. The gap between T1 and T2 without digits (12.5 % vs 12.1 %) is therefore within noise. Both are ~1.3 points behind MMS.

**Voice data:** 4,732 clips (6.84 h), kept from 5,502 recorded by the Common Voice speaker with the most material. Each model trained for ~1,210 epochs, ~22 h per model, and the exported ONNX model is 61 MB. The speaker's own clips score UTMOS 3.17, so the lost naturalness comes from the model, not the recordings.

## Reproducing

```bash
uv run guanani eval nmt flores --direction es-gn --model models/e7/es-gn
uv run guanani eval stt cv --model models/w2
uv run guanani eval tts flores --model models/t2 --asr-model models/w2 --limit 200
uv run guanani --device cpu eval stt cv --limit 50     # CPU latency
```

Each run writes a JSON file to `results/` (metrics, latency and 20 samples) and appends a row to `results/SUMMARY.md`. The TTS WAV files go to `results/audio/` and are not versioned.
