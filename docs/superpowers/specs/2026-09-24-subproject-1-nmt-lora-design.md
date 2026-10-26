# Subproject 1 — LoRA fine-tuning of NLLB-1.3B for gn↔es

Date: 2026-09-24 · Background: `docs/superpowers/specs/2026-09-23-subproject-0-skeleton-eval-design.md`, `results/SUMMARY.md`

## Context

Subproject 0 found that the weakest component is **es→gn** translation: on FLORES+ devtest, the NLLB-1.3B base scores chrF++ 33.05 and BLEU 7.29. The gn→es base scores chrF++ 39.85. Jojajovai scores are ~14 points higher, probably because NLLB saw that domain, so FLORES+ is the reliable reference.

Constraints: everything runs locally on a shared RTX 3060 with ~7 GB of free VRAM. The user's other services take ~4.8 GB and are not touched. The cloud is left for a later phase if results justify it.

## Goal and success thresholds

Improve es→gn without degrading gn→es. Measured on **FLORES+ devtest**:

1. es→gn: **≥ +3 chrF++** over the base (≥ 36.05).
2. gn→es: **no worse** than the base (≥ 39.85).
3. Jopara input: on `jojajovai-anotado` gn→es, overall chrF++ **≥ the base** (1.3B base: 49.31).

If an experiment meets all three, we recommend moving to full fine-tuning in the cloud.

## Approach

A `guanani.train` module with three CLI subcommands (`guanani train prepare|backtranslate|lora`) on `transformers` + `peft`. The LoRA adapter is merged and a full model is saved, so `guanani eval --model models/<name>` works unchanged.

Two experiments:
- **E1:** LoRA on real parallel corpora.
- **E2:** E1 + back-translation.

## Data

### Real parallel data

| Source | Approx. size | Origin |
|---|---|---|
| Jojajovai train | 20,207 pairs | `data/jojajovai/jojajovai_all.csv` (`split == "train"`) |
| AmericasNLP 2021 train | ~26k pairs | GitHub `AmericasNLP/americasnlp2021`, `data/guarani-spanish/train.gn` / `train.es` |
| agustin-guarani-llm-splits train | to be checked | HF `guaran-ia/agustin-guarani-llm-splits` (gated, automatic approval). The schema is checked in the plan. Without access it is skipped and noted in `stats.json` |

### Cleaning (in this order)

1. Normalise: the gn side with `normalize`; the es side with NFC + whitespace collapsing.
2. Filter out pairs where:
   - either side is empty;
   - both sides are identical;
   - either side has more than 200 words;
   - the word-length ratio is above 3.
3. Deduplicate exact pairs (lower-cased).
4. **Decontaminate:** drop any pair whose gn or es side matches an evaluation sentence.
   - Evaluation set: FLORES+ dev and devtest (gn and es), Jojajovai test and annotated sample (gn and es), Common Voice test (gn).
   - Comparison key: `normalize` + lower case + no punctuation or repeated spaces.

### Back-translation

- **Source:** KuatIA (`guaran-ia/kuatia`, gated).
  - Only subsets that are probably formal register: `madlad400`, `culturax`, `fineweb-2`, `hplt-3`, `FinePDF`, `guarania-scraped`, `multi-wiki-qa`.
  - Always excluded: `flores-200`, `belele`, `jojajovai`, `commonvoice`, `americasnlp*`, `americasnli`, Twitter, affective or chat corpora, and the Bible.
- Split documents into sentences on `.!?` followed by a space.
- Keep sentences of 5–60 words, deduplicated and decontaminated with the same key.
- Sample 60,000 sentences with a fixed seed and translate them gn→es with the NLLB-1.3B base (beam 4), in resumable blocks of 1,000.

### Outputs

- `data/train/parallel.jsonl` — `{gn, es, source}`
- `data/train/bt.jsonl` — `{gn, es, source: "bt:<subset>"}` (synthetic es, real gn)
- `data/train/stats.json` — incoming pairs and those removed, by filter and by source

## Training

- **Base:** `facebook/nllb-200-distilled-1.3B`, frozen in bf16.
- **LoRA:** r=16, alpha=32, dropout 0.05; modules `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` (encoder and decoder).
- **Examples:** each real pair produces two examples (gn→es and es→gn) with `src_lang`/`tgt_lang` (`grn_Latn`/`spa_Latn`). Back-translated pairs produce **only** es→gn (synthetic es → real gn).
- **Data selection:** `--data parallel` (E1) or `--data parallel,bt` (E2).
- **Default hyperparameters**, all changeable from the CLI:
  - AdamW, lr 2e-4, cosine scheduler, 3 % warmup;
  - per-device batch 8 × accumulation 4 (effective 32), 2 epochs;
  - max length 200 tokens, gradient checkpointing, seed 42.
- **Checkpoint selection:** every 1,000 steps, compute chrF++ (greedy) on the first 300 FLORES+ dev sentences in both directions, and keep the checkpoint with the highest mean across directions.
- `--max-steps` gives short test runs and memory measurements; `--resume` continues from the last checkpoint.
- **Output:** `models/<name>/` with the merged fp16 model and `train_meta.json` (data used, hyperparameters, best step, dev scores). `models/` is not versioned.

Estimate on the 3060: E1 ~5–6 h, E2 ~8–9 h.

## Evaluation

With the existing harness, `guanani eval nmt <dataset> --direction <d> --model models/<name>`:

- FLORES+ devtest, gn→es and es→gn (decides the thresholds).
- `jojajovai-anotado` gn→es (Jopara threshold).
- `jojajovai` test in both directions (reference only, given the contamination caveat).

The rows are added to `results/SUMMARY.md` next to the base. The 20 samples in each JSON allow a by-eye comparison of base / E1 / E2.

## Error handling

- No access to KuatIA or agustin → `DataMissing` with the URL for accepting the conditions. agustin is optional; KuatIA is required only for `backtranslate`.
- `backtranslate` writes blocks of 1,000 and resumes from the last complete block.
- OOM during training → a message suggesting a lower `--batch-size` (accumulation keeps the effective batch).
- If any evaluation sentence remains in the training data after decontamination, `prepare` fails with an error, not a warning.

## Tests (pytest)

- Cleaning: empty sides, identical sides, more than 200 words, ratio > 3, deduplication.
- Decontamination: a test sentence with different punctuation, case or apostrophe variant is removed.
- Sentence splitting and the KuatIA allow-list (excluded sources never get in).
- Example building: one real pair → two examples with the correct codes; back-translation → es→gn only.
- Slow (`slow`): 20 LoRA steps with NLLB-600M on 50 pairs; the merged model loads with `nmt.translate` and translates.

## Definition of done

1. `guanani train prepare`, `backtranslate` and `lora` work end to end.
2. E1 and E2 trained and evaluated, with rows in `results/SUMMARY.md`.
3. A conclusions paragraph: whether each experiment meets the three thresholds, what back-translation contributes, and whether moving to the cloud is recommended.

## Out of scope

Cloud, full fine-tuning, CTranslate2, publishing the model on HF, human evaluation, English/French, a formal/Jopara classifier.
