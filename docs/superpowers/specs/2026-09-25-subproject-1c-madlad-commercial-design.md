# Subproject 1c — Commercially licensed translation: MADLAD-400 and full fine-tuning

Date: 2026-09-25 · Background: specs for subprojects 1 and 1b, `results/SUMMARY.md`

## Context and change of direction

The product will be **used commercially**. Consequences:
- NLLB-200 (all sizes) and MMS (ASR/TTS) are CC-BY-NC-4.0, so **they cannot be used in the product**. E1–E4 (LoRA on NLLB-1.3B) remain as research and a quality reference.
- The `bt2` back-translation was generated with E2 (NLLB), so it is not used to train the commercial model.
- Permissively licensed candidates that include Guarani (`gn`): **MADLAD-400-3B-MT** and **MADLAD-400-7B-MT** (Apache 2.0).
- This document is a technical licence analysis, not legal advice.

Serving: the service will run on a GPU, even a small one, so ~3B models in INT8 are viable.

Lessons so far: with NLLB-1.3B + LoRA, scaling and filtering the back-translation barely improved es→gn (E3 dev es→gn 32.46 vs 32.17 for E2), and the dev curve plateaus early. So we now try full fine-tuning and a different base model.

## Goal and thresholds

Same absolute thresholds (FLORES+ devtest, except the third), measured on commercially usable models:
1. es→gn chrF++ ≥ 36.05.
2. gn→es chrF++ ≥ 39.85.
3. `jojajovai-anotado` gn→es chrF++ ≥ 49.31.

BLEU and each model's licence are also reported.

## Approach

Extend the existing `nmt` and training code with model-family detection (NLLB / MADLAD) and a full fine-tuning mode. Experiments:
1. **Screening** without training: MADLAD-3B and MADLAD-7B on FLORES devtest. The NLLB-1.3B base, already measured, is the reference.
2. **E5:** full fine-tuning of MADLAD-3B.
3. **E6:** LoRA on MADLAD-7B, only if, in the screening, 7B beats 3B by ≥ 2 chrF++ on es→gn and fits in 24 GB in bf16.

## MADLAD support

- The family is detected from the model config: `model_type == "t5"` → MADLAD; otherwise NLLB. This works with fine-tuned models in `models/…`.
- MADLAD: the direction is given by the prefix `<2gn> ` / `<2es> ` on the input text, with no `src_lang` or `forced_bos_token_id`.
- MADLAD loads in bf16 on GPU, because T5 is unstable in fp16.
- This affects `nmt.translate`, the tokenisation of training examples and dev evaluation. Normalisation, metrics, reports and the evaluation CLI do not change.

## Data licences

- A new file, `docs/data-licenses.md`, holds a table with each source's licence, link, commercial use (yes/no/unknown) and conditions.
  - Sources: Jojajovai, AmericasNLP 2021, the 7 KuatIA sources (`madlad400`, `culturax`, `fineweb-2`, `hplt-3`, `FinePDF`, `guarania-scraped`, `multi-wiki-qa`) and Wikipedia.
  - Evaluation only: FLORES+ and Common Voice. Tools: GlotLID and COMET.
- Rule: a non-commercial source, or one with an unknown licence, is excluded from commercial training.
- The table has a machine-readable column with the `source` identifier used in the data (`jojajovai`, `americasnlp`, `bt:madlad400`, …) and whether it is approved.

## `bt3` back-translation

- Reuses the `bt2` sample (499,420 Guarani sentences already filtered with GlotLID): `guanani train backtranslate --name bt3 --from bt2 --model google/madlad400-3b-mt --sample --batch-size 64`.
- `--from X` copies `data/train/X/source.jsonl` without redoing the selection and without modifying `X`.
- Top-k = 10 sampling, as in `bt2`. The result goes to `data/train/bt3/`.
- The licence filter is applied at training time, not at back-translation time.

## Training

- The subcommand becomes `guanani train nmt`; `lora` is kept as an alias.
- The default mode is LoRA (as now). `--full` runs full fine-tuning with Adafactor, bf16 and gradient checkpointing. It saves the best full model directly in `models/<name>/`, with no adapter or merge step. Dev selection and `--resume` work as before.
- `--sources-allow docs/data-licenses.md`: only examples whose `source` is approved in the table are used.
- **E5:** `--base google/madlad400-3b-mt --full --data parallel,bt3 --upsample-parallel 4 --epochs 1 --lr 1e-4 --sources-allow docs/data-licenses.md`. Batch size and evaluation steps are set after a short memory test on the server.
- **E6:** `--base google/madlad400-7b-mt` with LoRA r=16 and the same data. If it does not fit in 24 GB in bf16, it is dropped and documented (no quantisation this round).

## Evaluation

The usual 5 evaluations per model. The conclusions include:
- a table of NLLB-1.3B base (non-commercial reference) / MADLAD-3B base / MADLAD-7B base / E5 / E6 against the thresholds, in chrF++ and BLEU, with a licence column;
- the best commercially usable model and its distance to the thresholds.

## Error handling

- `--from X` without `data/train/X/source.jsonl` → `DataMissing`.
- `--sources-allow` with a file that lacks the expected table → an error that describes the format.
- If the licence filter leaves no examples of some data type → an error, not an empty training run.

## Tests (pytest, no real models)

- Family detection by `model_type`.
- `<2gn>`/`<2es>` prefixes in `translate` and in MADLAD example tokenisation; NLLB unchanged.
- `--full` does not apply LoRA and saves the full model as the best checkpoint.
- `--from bt2` reuses the sample without modifying `bt2`.
- `--sources-allow` lets through only approved sources; table parsing.
- `train lora` still works as an alias of `train nmt`.

## Definition of done

1. `docs/data-licenses.md` in the repository.
2. Screening measured.
3. `bt3` generated.
4. E5 trained and evaluated; E6 trained and evaluated, or dropped with a reason.
5. Conclusions: the best commercially usable model and its distance to the thresholds.

## Out of scope

Quantisation, distillation, commercial ASR/TTS (they have their own subprojects), legal advice.
