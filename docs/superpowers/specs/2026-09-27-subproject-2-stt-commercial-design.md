# Subproject 2 — Commercially licensed speech recognition (STT)

Date: 2026-09-27 · Background: subproject 0 spec, `results/SUMMARY.md`, `docs/data-licenses.md`

## Context

The product is commercial. MMS-1B-all (CC-BY-NC-4.0), the subproject 0 baseline, cannot be used. It scores CER 5.8 % and WER 41.6 % on the Common Voice 17 gn test. Permissively licensed candidates:
- **Omnilingual ASR** (Meta, Apache 2.0): includes `gug_Latn` (Paraguayan Guarani) without fine-tuning. It depends on `fairseq2` (≤ 0.6), which is incompatible with the project's torch.
- **Whisper large-v3-turbo** (MIT): does not include Guarani; it is fine-tuned by reusing the `es` label.

Uses: first short dictated sentences (speech → text → translation), then long audio (voice notes, recordings, Jopara).

This document is a technical licence analysis, not legal advice.

## Goal and thresholds

On the Common Voice 17 gn test (855 clips), with a commercially usable model:
- **Minimum:** match MMS: CER ≤ 5.8 % and WER ≤ 41.6 %.
- **Target:** WER ≤ 35 %.

Omnilingual and MMS may have seen Common Voice during training; the results say so.

## Data

Common Voice 17 gn (CC0), mirror `fsicoli/common_voice_17_0`:
- **Validated splits:** `train` (1,571 clips) for training, `dev` (360) for picking the checkpoint, `test` (855) for evaluation only.
- **`other`** (~18,800 unvalidated clips, ~25 h): a separate experiment; it is used only if it improves dev.
- **Decontamination:** any sentence whose normalised key (`decont_key`) is in test is dropped from training and dev.
- Empty or unreadable audio is dropped and counted in the statistics.
- `scripts/download.sh` also downloads `train`, `dev` and `other`.
- `docs/data-licenses.md`: Common Voice becomes a training source (CC0); Omnilingual ASR (Apache 2.0) and Whisper (MIT) are added.

## Evaluation

- CER and WER with `guanani eval stt` on `cv` test, the same as MMS.
  - Text is normalised before comparison: NFC, lower case, no punctuation, unified apostrophes and nasals (`normalize.py`).
  - One function is shared by evaluation and training.
- The best checkpoint is picked by CER on dev.
- **Long audio (a check, not the main metric):** test clips are joined into 1–2 min recordings, split on silences (`librosa.effects.split`) and transcribed, and the WER is reported.

## Experiments

All run on the remote server (RTX PRO 5000, 24 GB).

| Id | Model | Data | Question |
|---|---|---|---|
| S0 | Omnilingual CTC-300M, CTC-1B, CTC-3B, LLM-1B, not fine-tuned (`gug_Latn`) | — | Commercial baseline and size |
| S1 | Omnilingual CTC, the best size that fits on a small GPU, fine-tuned | CV train | Minimal fine-tuning |
| S2 | S1 | CV train + `other` | Does unvalidated audio help? |
| W1 | Whisper large-v3-turbo, full fine-tuning, `es` label, bf16 | CV train | Whisper alternative |
| W2 | W1 | CV train + `other` | Only if S2 improves on S1 |

"Fits on a small GPU" means inference in ≤ 8 GB of VRAM. The 7B variants are excluded, except as a reference if S0 needs them.

## Code

- **`src/guanani/stt.py`:** `transcribe` picks a backend from the model name or the config in `models/<name>`:
  - `mms`: the current one (`facebook/mms-*`).
  - `whisper`: `transformers` (`openai/whisper-*`, or fine-tunes with `model_type == "whisper"`), with `language="es"` and `task="transcribe"`.
  - `omnilingual`: `facebook/omniASR-*`, or fine-tunes from that recipe.
    - It is not imported directly. Instead it is called via `subprocess` in `envs/omnilingual/`, a separate uv environment with fairseq2 and `omnilingual-asr`.
    - It takes a list of audio paths and returns transcriptions as jsonl.
    - If the environment is missing: `RuntimeError` with the command to create it.
  - Long audio: `transcribe_long(audio)` splits on silences into chunks of ≤ 30 s and concatenates the results.
  - OOM during batch inference: the batch is split in half.
- **`src/guanani/train/stt_data.py`:** loads the Common Voice TSVs, decontaminates against test, normalises text and writes statistics. It exports to the parquet format of the Omnilingual recipe.
- **`src/guanani/train/whisper.py`:** fine-tuning with `Seq2SeqTrainer`.
  - bf16, Adafactor; clips longer than 30 s are dropped.
  - The best checkpoint is picked by CER on dev, including an evaluation at step 0.
  - `--resume`; the output directory is wiped unless resuming.
  - Output in `models/<name>/` with `train_meta.json`.
- **Omnilingual fine-tuning:** the official fairseq2 recipe inside `envs/omnilingual/`, launched with `scripts/omni_finetune.sh` from the `stt_data.py` data.
- **CLI:** `guanani train stt --name <id> --backend whisper|omnilingual --data train|train+other ...`. Data or configuration errors → exit code 2.
- **Licences:** training accepts only the base models `openai/whisper-*` and `facebook/omniASR-*`, and Common Voice data.

Out of scope: service or API and INT8 (subproject 5), a KenLM language model for CTC, advanced VAD.

## Tests

- **Fast:**
  - per-sentence decontamination;
  - reference normalisation;
  - reading Common Voice TSVs;
  - backend choice by name or config;
  - export to the Omnilingual parquet;
  - silence splitting on synthetic audio;
  - rejection of disallowed base models.
- **Slow (`-m slow`):**
  - fine-tuning Whisper tiny for a few steps on 8 clips (saves the best checkpoint and transcribes);
  - transcribing a clip with Omnilingual CTC-300M through the separate environment.

## Deliverables

- The code and tests above.
- Rows for S0–S2 and W1–W2 in `results/SUMMARY.md`, and the subproject 2 conclusions with each model's licence and a note on possible contamination.
- The best commercial STT model in `models/` on the remote server.
