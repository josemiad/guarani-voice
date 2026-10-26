# Subproject 0 — Skeleton with base models + evaluation harness

Date: 2026-09-23 · Reference document: `Guarani LLM and TTS Development.md`

## Context and project decisions

- **Purpose:** a public tool for Guarani speakers and learners. Success is perceived quality.
- **Current language scope:** **Guarani ↔ Spanish** only. English and French come in a later phase, pivoting through Spanish.
- **Register:** input can be formal Guarani or Jopara; Guarani output is formal/standard.
- **Compute:** a local RTX 3060 12 GB for iterating, plus occasional cloud GPUs for final training runs. Unrestricted cloud use (larger models) is left for phase 2, once NMT/STT/TTS work.
- **Deployment:** a private VPS (to be defined in subproject 5).
- **Human evaluation:** not available for now, so only automatic metrics. Human evaluation is required before public launch.

### Project breakdown

| # | Subproject | Base | Improvement |
|---|---|---|---|
| 0 | Skeleton + evaluation (**this document**) | — | — |
| 1 | NMT gn↔es | NLLB-600M / 1.3B base | Fine-tuning on Jojajovai |
| 2 | STT gn | MMS-1B-all (`grn`) | Fine-tuning on filtered Common Voice |
| 3 | TTS gn | MMS-TTS-grn | eSpeak-NG G2P + Piper with a recorded voice |
| 4 | es↔en/fr pivot | — | later phase |
| 5 | Service (API + web on a VPS) | — | — |

## Goal

Get speech → text → translation → speech working end to end with pre-trained models, with reproducible baseline figures per component to decide which subproject to tackle first.

## Structure

```
pyproject.toml            # uv; torch, transformers, sentencepiece, librosa, soundfile, sacrebleu, jiwer, unbabel-comet, huggingface_hub
src/guanani/
  normalize.py            # normalize(text) -> str
  nmt.py                  # translate(texts, src, tgt, model=...) -> list[str]
  stt.py                  # transcribe(wavs, model=...) -> list[str]
  tts.py                  # speak(text, model=...) -> (audio, sr)
  data.py                 # loads the evaluation sets
  eval.py                 # runs a component against a set and writes a report
  cli.py                  # guanani translate|transcribe|speak|pipeline|eval
scripts/download.sh       # downloads public datasets; documents the ones that need an account
tests/
data/                     # not versioned
results/                  # versioned reports; results/audio/ not versioned
```

## Components

- **normalize:** applied to all Guarani text: input, output and references.
  - Unicode NFC.
  - Apostrophe variants (’ ‘ ´ \` ʼ ꞌ ′ ‚) → the canonical *puso* `'` (U+0027).
  - Legacy nasal spellings seen in Jojajovai → tilde: â ê î ô û ŷ ÿ ä ë ï ö → ã ẽ ĩ õ ũ ỹ, and ĝ ḡ ğ ġ → g̃. `ü` is kept for loanwords such as *güe*.
  - Removes zero-width spaces and collapses whitespace.
- **nmt:** NLLB with the codes `grn_Latn` / `spa_Latn`, in batches. The model is selectable (`--model`). Baselines: `facebook/nllb-200-distilled-600M` and `facebook/nllb-200-distilled-1.3B` (fp16 on GPU).
- **stt:** `facebook/mms-1b-all` with the `grn` adapter, 16 kHz audio.
- **tts:** `facebook/mms-tts-grn` (VITS, 16 kHz).
- **pipeline:** gn speech → stt → normalize → nmt gn→es; and es text → nmt es→gn → tts.

Each component takes the model id as a parameter, so the fine-tuned models from subprojects 1–3 plug in without code changes and are compared with the same `eval`.

## Evaluation data

| Set | Use | Source |
|---|---|---|
| FLORES+ devtest (`gug_Latn`, `spa_Latn`) | NMT, general domain | HF `openlanguagedata/flores_plus` (gated) |
| Jojajovai test (5,334 pairs) | NMT, overall | GitHub `pln-fing-udelar/jojajovai`, `jojajovai_all.csv` |
| Jojajovai annotated sample (400 pairs) | NMT, broken down by dialect (guarani / jopara / jehe'a / other) | same repo, `jojajovai_sample_annotations.csv` |
| Common Voice gn test | STT | Mozilla Data Collective (account, manual download) |
| FLORES gn sentences | TTS round trip | reuses FLORES |

The exact sources are checked at the start of implementation. If any has changed, the loader adapts, not the design.

## Metrics

- **NMT** (gn→es and es→gn separately): chrF++ (primary), BLEU (reference), COMET wmt22. COMET into Guarani is reported as **indicative** only, because XLM-R's coverage of Guarani is weak.
- **STT:** WER and CER, with hypothesis and reference normalised the same way (normalize + lower case + no punctuation except the *puso*).
- **TTS:** round-trip CER (speak → transcribe → compare with the text) + 10 sample WAVs.
- **All:** latency per sentence (or RTF for audio) and device. `--device cpu` measures the VPS scenario.

## Reports

Each run writes `results/<date>_<component>_<model>_<dataset>.json` with metrics, configuration and 20 samples (source/reference/hypothesis), and appends a row to `results/SUMMARY.md`. `--limit N` gives quick test runs.

## Error handling

- Missing dataset → a clear error naming the expected file and how to get it.
- Audio at another sample rate → resampled to 16 kHz on load. Empty or corrupt clips are skipped and counted as `skipped` in the report.
- No GPU → CPU with a warning; the device is recorded in the report.
- No retries and no custom cache (the Hugging Face cache is enough).

## Tests (pytest)

- `normalize`: apostrophe variants → U+0027; NFD → NFC; combined vs precomposed `ã`; `g̃` (which has no precomposed form) is preserved.
- Metrics: `eval` computes chrF++/WER correctly on toy pairs, without models.
- Model smoke tests marked `slow` (excluded by default): one sentence per component.

## Definition of done

1. `guanani translate`, `transcribe`, `speak` and `pipeline` work locally with the base models.
2. `results/SUMMARY.md` contains baseline figures for:
   - NMT (600M, 1.3B) × (gn→es, es→gn) × (FLORES, Jojajovai test, annotated sample by dialect);
   - STT on Common Voice;
   - TTS round trip;
   - latency on GPU and CPU.
3. A conclusions paragraph in `SUMMARY.md` names the weakest component and the next subproject.

## Licences

NLLB-200 and MMS (ASR and TTS) are CC-BY-NC-4.0, which is acceptable for a non-commercial public tool. Any commercial use requires replacing them.

## Out of scope

Fine-tuning, API/server, web interface, English/French, human evaluation, CTranslate2.
