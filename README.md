# Guanani

Open-source Guarani ↔ Spanish machine translation, Guarani speech recognition and Guarani speech synthesis. The project includes training pipelines, an evaluation harness and every measured result.

Guarani is spoken by about 6 million people in Paraguay and its neighbouring countries, yet it has few language tools. Guanani fine-tunes open models with public data and measures them on standard benchmarks. The work is organised as subprojects: 0 is the baseline and evaluation harness, 1–1d cover translation, 2 speech recognition and 3 speech synthesis.

## Results at a glance

| Component | Best model with a commercial-use licence | Result | Threshold | Status |
|---|---|---|---|---|
| Translation es→gn | MADLAD-400 3B base | FLORES+ chrF++ 34.24 | ≥ 36.05 | ❌ 1.8 short |
| Translation gn→es | E5 (fine-tuned MADLAD-3B) | FLORES+ chrF++ 39.49; Jopara 52.69 | ≥ 39.85; ≥ 49.31 | ❌ 0.36 short / ✅ |
| Speech → text | W2 (fine-tuned Whisper large-v3-turbo) | Common Voice WER 21.3 %, CER 3.6 % | ≤ 41.6 %, ≤ 5.8 % | ✅ |
| Text → speech | T2 (Piper trained from scratch) | Round-trip CER 12.1 %, UTMOS 2.57, CPU RTF 0.035 | ≤ 13.9 %, ≥ 3.18, ≤ 0.2 | speed only ✅ |

The thresholds come from the subproject specs. For translation, es→gn must reach the NLLB-1.3B base + 3 chrF++, and gn→es and Jopara must not fall below that base. For speech recognition, the model must at least match MMS-1B-all. For speech synthesis, it must at least match MMS-TTS on intelligibility and naturalness, and run in real time on CPU.

## All results

### Translation: FLORES+ devtest (1,012 sentences)

| Model | Licence | es→gn chrF++ | es→gn BLEU | gn→es chrF++ | gn→es BLEU | gn→es COMET | Jopara gn→es chrF++ (400) |
|---|---|---|---|---|---|---|---|
| NLLB-200 600M | CC-BY-NC | 31.75 | 6.75 | 37.96 | 13.90 | 0.649 | 46.69 |
| NLLB-200 1.3B | CC-BY-NC | 33.05 | 7.29 | 39.85 | 15.71 | 0.689 | 49.31 |
| E1: NLLB-1.3B + LoRA, real pairs | CC-BY-NC | 32.40 | 7.12 | 40.32 | 16.00 | 0.689 | 50.51 |
| E2: E1 + 60k back-translated | CC-BY-NC | 33.54 | 6.69 | **40.63** | 16.16 | 0.690 | 50.88 |
| E3: + 500k back-translated, GlotLID | CC-BY-NC | 33.67 | 7.09 | 40.52 | 16.10 | 0.690 | 50.76 |
| E4: E3 + `[BT]` tag | CC-BY-NC | 33.34 | 7.23 | 40.58 | **16.19** | 0.690 | 51.01 |
| MADLAD-400 3B | Apache 2.0 | 34.12 | 7.02 | 38.13 | 14.21 | 0.687 | 42.39 |
| MADLAD-400 7B | Apache 2.0 | 29.10 | 4.88 | 38.17 | 14.50 | 0.694 | — |
| E5: MADLAD-3B, full fine-tuning | Apache 2.0 | 32.00 | 5.74 | 39.49 | 15.46 | **0.699** | **52.69** |
| E7: MADLAD-3B + LoRA, best per direction | Apache 2.0 | **34.24** | 6.99 | 39.40 | 15.35 | 0.694 | 51.45 |
| Google Translate (200 sentences, scored in blocks) | — | — | — | 40.41 | — | — | — |

### Speech recognition: Common Voice 17 Guarani test (855 clips)

| Model | Training data | Licence | WER | CER | GPU RTF |
|---|---|---|---|---|---|
| MMS-1B-all | — | CC-BY-NC | 41.6 % | 5.8 % | 0.024 |
| Omnilingual CTC-300M v2 | — | Apache 2.0 | 62.9 % | 13.5 % | 0.003 |
| Omnilingual CTC-1B v2 | — | Apache 2.0 | 50.2 % | 10.3 % | 0.014 |
| Omnilingual CTC-3B v2 | — | Apache 2.0 | 38.9 % | 7.5 % | 0.502 |
| Omnilingual LLM-1B v2 | — | Apache 2.0 | 36.4 % | 6.8 % | 0.076 |
| W1: Whisper large-v3-turbo fine-tuned | CV train (1.7 h) | MIT | 31.9 % | 5.4 % | 0.013 |
| S1: Omnilingual CTC-1B fine-tuned | CV train | Apache 2.0 | 30.7 % | 4.4 % | 0.005 |
| S2b: Omnilingual CTC-1B fine-tuned | CV train + other (11 h) | Apache 2.0 | 23.1 % | **3.2 %** | 0.005 |
| **W2: Whisper large-v3-turbo fine-tuned** | CV train + other | MIT | **21.3 %** | 3.6 % | 0.013 |

### Speech synthesis: 200 FLORES+ sentences, round trip through W2

| Voice | Licence | CER | CER without digits (141) | UTMOS | CPU RTF |
|---|---|---|---|---|---|
| MMS-TTS-grn | CC-BY-NC | **13.9 %** | **10.9 %** | **3.18** | 0.25 |
| T1: Piper, eSpeak-NG phonemes | commercial | 17.0 % | 12.5 % | 2.54 | 0.035 |
| T2: Piper, letters | commercial | 39.0 %¹ | 12.1 % | 2.57 | **0.035** |

¹ T2 does not read digits and rejects the 59 sentences that contain them.

[docs/results.md](docs/results.md) has the methodology, the Jojajovai scores, CPU latency, long-audio checks, training times and caveats. [results/SUMMARY.md](results/SUMMARY.md) has the experiment log, and each run's raw report is a JSON file in `results/`.

## Usage

```bash
uv sync
uv run hf auth login             # FLORES+ and Common Voice are gated datasets
./scripts/download.sh            # Jojajovai + FLORES+ + Common Voice 17 (gn)
uv run guanani translate "Mba'éichapa reiko?"
uv run guanani speak "Mba'éichapa" -o hola.wav
uv run guanani transcribe hola.wav
uv run guanani pipeline --text "Buenos días" -o saludo.wav
uv run guanani eval nmt flores --direction gn-es
```

Without `--model`, commands use the NLLB-200 600M, MMS-1B-all and MMS-TTS base models. To use a model you have trained, pass its directory with `--model`, for example `models/e5`, `models/e7/es-gn`, `models/w2` or `models/t2`. Trained weights are not included in the repository.

**Training:** use `guanani train prepare|backtranslate|nmt|stt|omni-export|omni-pick|tts-prep|tts-pick` together with the scripts in `scripts/`. Omnilingual and Piper run in separate environments under `envs/`.

**Tests:** `uv run python -m pytest` runs the fast tests; add `-m slow` to run the tests that load real models.

## Documentation

- [docs/results.md](docs/results.md): every measured result and the methodology.
- [docs/data-licenses.md](docs/data-licenses.md): licence audit of each dataset and model.
- [docs/superpowers/](docs/superpowers/): design spec and implementation plan for each subproject.
- [Guarani LLM and TTS Development.md](<Guarani LLM and TTS Development.md>): the initial research survey.

## Licence

The code and documentation in this repository are released into the public domain under [the Unlicense](LICENSE). You may use, modify and redistribute them for any purpose, with no conditions.

The Unlicense covers only this repository. Third-party models and datasets keep their own licences (see [docs/data-licenses.md](docs/data-licenses.md)):

- **NLLB-200 and MMS** (the CLI's default models and the E1–E4 fine-tunes) are CC-BY-NC-4.0: non-commercial use only.
- **MADLAD-400** (Apache 2.0), **Whisper** (MIT) and **Omnilingual ASR** (Apache 2.0) allow commercial use, with their own notice requirements.
- **Piper's training code** is GPL-3.0. It is used as an external tool and is not included here.
- **Training data** carries attribution or share-alike terms: Jojajovai (MIT), Guarani Wikipedia (CC BY-SA), and KuatIA's sources. The prototype TTS voice comes from a Common Voice (CC0) donor.
