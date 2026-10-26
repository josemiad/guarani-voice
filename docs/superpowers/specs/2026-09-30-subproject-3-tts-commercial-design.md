# Subproject 3 — Commercially licensed speech synthesis (TTS)

Date: 2026-09-30 · Background: subproject 0 spec, `results/SUMMARY.md` (subproject 2: STT W2), `docs/data-licenses.md`

## Context

The product is commercial. MMS-TTS-grn (CC-BY-NC-4.0), the subproject 0 baseline, cannot be used. It scores a round-trip CER of 11.7 % on 200 FLORES+ sentences with MMS as the recogniser.

Fine-tuning MMS-TTS would inherit its non-commercial licence, and most published Piper voices were fine-tuned from "lessac" (a research-only licence). **The only clean route is training from scratch** on our own data.

- **Phase A (this spec):** a prototype with the voice of the Common Voice gn (CC0) speaker with the most material, to validate the whole system.
- **Phase B (future, out of scope):** a professional voice actor, recorded or licensed. Only the data changes.

Tool: Piper (`OHF-Voice/piper1-gpl`, VITS).
- Its code is GPL-3.0. We run it on a server without distributing it, so we are not obliged to publish our code, and the trained voices are not GPL.
- It exports to ONNX and synthesises in real time on CPU.

This document is a technical licence analysis, not legal advice.

## Goal and thresholds

On the first 200 FLORES+ devtest sentences (the same as the subproject 0 baseline), with a commercially usable model:
- **Intelligibility:** round-trip CER (synthesis → transcription with W2) ≤ that of MMS-TTS measured with the same recogniser (T0).
- **Naturalness:** UTMOS ≥ that of T0.
- **Speed:** CPU RTF ≤ 0.2.

10 WAVs per model are kept for listening. Intelligibility and naturalness are not measured with listeners.

## Data

The Common Voice 17 gn speaker with the most clips (~5,500 clips, ~6.4 h raw, mostly in `other`), identified by `client_id`. Preparation lives in `src/guanani/train/tts_data.py`:

1. **Content filter:** each clip is transcribed with W2 (`models/w2`) and dropped if its CER against the text is > 10 %.
2. **Audio:**
   - trim leading and trailing silence;
   - normalise volume to −20 dBFS;
   - resample to 22,050 Hz mono WAV;
   - drop clips shorter than 1 s or longer than 15 s after trimming;
   - drop clipped audio (≥ 0.1 % of samples with |x| ≥ 0.999).
3. **Text:** `normalize()`. Sentences with digits are dropped, because there is no reliable way to read numbers in Guarani, and so are empty sentences.
4. **Decontamination:** sentences whose `decont_key` is in FLORES+ dev or devtest are dropped.
5. **Split:** 97 % training, 3 % Piper validation, with no sentence shared between them.
6. **Output:** `data/train/tts/<name>/{wavs/, metadata.csv, val.csv, stats.json}`.
   - `metadata.csv` uses `|` as the separator, in Piper's format (`file.wav|text`).
   - `stats.json` counts incoming clips, discards by reason and final hours.

An unreadable or empty clip, or one with no transcription, is dropped and counted; it never aborts preparation.

**Plan B (T3):** if fewer than 3 usable hours remain, or T1 and T2 miss the threshold:
1. pre-train a multi-speaker model on the Common Voice speakers with ≥ 1 h each (~15 h raw, same filter);
2. then fine-tune it on the main speaker.

`docs/data-licenses.md` changes:
- The prototype voice belongs to a Common Voice donor (CC0) who has not expressly consented to a synthesiser. It is marked as a prototype, to be replaced by a professional voice actor before launch.
- Piper (GPL-3.0, server-side training only), `piper-tts` (inference) and UTMOS (evaluation only) are added.

## Experiments

On the remote server (RTX PRO 5000, 24 GB); T1 and T2 can run at the same time.

| Id | Model | Input | Question |
|---|---|---|---|
| T0 | MMS-TTS-grn (non-commercial) | — | Baseline with W2 as the recogniser |
| T1 | Piper *medium*, 22,050 Hz, from scratch | eSpeak-NG `gn` phonemes | Commercial voice with phonemes |
| T2 | Piper *medium*, 22,050 Hz, from scratch | letters (`phoneme_type text`, NFD) | Commercial voice without eSpeak |
| T3 | Multi-speaker CV Piper → fine-tuned to the main speaker | the better of T1/T2 | Only if plan B applies |

Budget: ~1 GPU day per experiment. The best checkpoint is picked by round-trip CER (W2) on 100 FLORES+ dev sentences; lower is better.

## Code

- **`envs/piper/`:** a separate uv environment with `piper1-gpl[train]` pinned to a commit, used only for training and ONNX export. It includes the `monotonic_align` build.
- **`src/guanani/tts.py`:** `speak(text, model, device) -> (audio, sr)` picks a backend:
  - `piper` if `model` is a directory with `*.onnx` and `*.onnx.json` (inference with `piper-tts` in the main environment);
  - `mms` otherwise.
  - Empty text → `ValueError`. Missing voice → an error with the command to get it.
- **`src/guanani/train/tts_data.py`:** data preparation. CLI: `guanani train tts-prep --name <voice> [--speaker <client_id>]`, defaulting to the speaker with the most clips.
- **`scripts/piper_train.sh <name> <data> espeak|text [--resume]`:** training in `envs/piper`. The output directory is wiped unless `--resume`.
- **`guanani train tts-pick --name <id> --run-dir <dir>`:**
  - exports each checkpoint to ONNX (in `envs/piper`);
  - measures round-trip CER on dev;
  - leaves the best one in `models/<id>/` (ONNX + JSON + `train_meta.json`).
- **`guanani eval tts`:** gains the `--asr-model` option (the round-trip recogniser; defaults to the current one) and the UTMOS metric. Data or configuration errors → exit code 2.

Out of scope: a professional voice actor (phase B), reading numbers, prosody or speed control, multiple voices in the product, the service (subproject 5).

## Tests

- **Fast:**
  - CER and duration filters;
  - clipping detection;
  - dropping digits;
  - decontamination against FLORES;
  - split with no shared sentences;
  - `metadata.csv` format;
  - backend choice by directory;
  - picking the best checkpoint (lowest CER);
  - an unreadable clip is dropped and counted.
- **Slow (`-m slow`, on the server):** train Piper for a few steps on 20 clips, export to ONNX and synthesise a sentence with `speak`.

## Deliverables

- The code and tests above.
- Rows for T0–T2 (and T3 if applicable) in `results/SUMMARY.md`, and the subproject 3 conclusions with each model's licence and audio samples.
- The best commercial voice in `models/` (ONNX), also copied to the local machine.
