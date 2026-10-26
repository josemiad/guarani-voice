# Subproject 1b — Scaled back-translation, GlotLID and tagged BT

Date: 2026-09-25 · Background: `docs/superpowers/specs/2026-09-24-subproject-1-nmt-lora-design.md`, `results/SUMMARY.md`

## Context

In subproject 1 no experiment met the main threshold. E2 (22,095 real pairs + 60,000 back-translated sentences) reached es→gn chrF++ 33.54 against 33.05 for the base, and its es→gn BLEU was below the base. Back-translation added +1.1 chrF++ (E2 − E1), and training plateaued early. The limit seems to be the quantity and quality of target-side Guarani.

Facts checked for this round:
- The 7 allow-listed KuatIA sources give 550,401 unique clean sentences under the current filters, so 500,000 is reachable without new sources.
- Guarani Wikipedia: 6,033 articles, ~1 M words.
- GlotLID (`cis-lmu/glotlid`, fastText, Apache 2.0) tells `gug` (Paraguayan Guarani) apart from `gun` (Mbyá), `gui`, `nhd`, `gnw`, `urb`, etc.
- Training runs on the user's 24 GB RTX PRO 5000 Blackwell (`user@gpu-server`, repo in `~/Guanani`): LoRA 1.3B at ~1.23 steps/s with batch 16 × 2.

## Goal and thresholds

Same thresholds as subproject 1, on FLORES+ devtest except the third:
1. es→gn chrF++ ≥ 36.05.
2. gn→es chrF++ ≥ 39.85.
3. `jojajovai-anotado` gn→es chrF++ ≥ 49.31.

The conclusions also report BLEU, so as not to depend on a single metric.

## Approach

Extend the existing pipeline (`guanani train …`) and run two experiments on the same data:
- **E3:** real pairs ×4 + 500,000 back-translated es→gn examples.
- **E4:** same as E3, with the tag `[BT] ` prepended to the synthetic Spanish (tagged back-translation).

## Data

### Variety filter

- GlotLID `model_v3.bin` via `fasttext`. A sentence passes if its top-1 label is `gug_Latn` or `grn_Latn` with probability ≥ the threshold.
- The threshold is calibrated as the highest value that lets through ≥ 95 % of the gn sides of `data/train/parallel.jsonl` (real Paraguayan Guarani), and is recorded in `stats.json`.
- Two rules from the current heuristic stay: Spanish stop-word share ≤ 25 %, and share of words with digits ≤ 20 %. GlotLID replaces the spelling rule (`ɨʉwq`, `nh`, `kuery`), which is removed.
- Length 5–60 words, deduplication and decontamination against the evaluation sets all stay.

### Guarani Wikipedia

- Dataset `wikimedia/wikipedia`, Guarani configuration (`*.gn`), as plain text. `scripts/download.sh` downloads it to `data/wikipedia/`. The exact configuration is checked in the plan.
- It goes through the same stages: sentence splitting, filters and decontamination.

### Composition of `bt2`

- All valid Wikipedia sentences, topped up to 500,000 with a sample from KuatIA (the 7 allow-listed sources). Fixed seed.
- Each sentence keeps its source: `bt:wikipedia`, `bt:madlad400`, etc.

### Generation

- `nmt.translate` gains `sample: bool = False`. With `sample=True` it generates with `do_sample=True`, `top_k=10`, `num_beams=1` and a fixed seed.
- `guanani train backtranslate --name bt2 --model models/e2 --sample --batch-size 64 --n 500000`.
- Everything goes under `data/train/bt2/`: `source.jsonl`, `parts/`, `bt2.jsonl`, `stats.json`. `data/train/bt.jsonl` (E2) is left untouched.
- Resumable in blocks of 1,000. The sample and block size are saved in `bt2/meta.json`. If the existing sample or block size does not match the blocks already generated, it fails with an error instead of mixing them.

## Training

- `DATA_FILES` is generalised:
  - `parallel` → `data/train/parallel.jsonl`;
  - `bt` → `data/train/bt.jsonl`;
  - `btN` (e.g. `bt2`) → `data/train/btN/btN.jsonl`.
- `--upsample-parallel N` (default 1) repeats the real pairs N times before building examples. E3 and E4 use 4.
- `--bt-tag` prepends `[BT] ` to the es side of back-translated examples. The tag is not used at inference.
- Training starts from the NLLB-1.3B base, with the same LoRA (r=16) and hyperparameters as subproject 1:
  - batch 16 × 2;
  - **1 epoch** (~21,000 steps, ~4.8 h per experiment);
  - dev evaluation every 2,000 steps.

## Execution

On the user's server:
1. sync the code with `rsync`;
2. download the new data;
3. generate `bt2` (~2 h);
4. train E3 and E4 back to back with `setsid nohup`;
5. evaluate there and bring `results/` back with `rsync`.

The models stay on the server.

## Evaluation

The usual 5 evaluations per model: FLORES+ devtest gn→es and es→gn, `jojajovai-anotado` gn→es, and `jojajovai` test gn→es and es→gn.

The conclusions include:
- a table of base / E1 / E2 / E3 / E4 against the thresholds, in chrF++ and BLEU;
- the effect of the tag (E4 − E3);
- the effect of scaling and filtering the back-translation (E3 − E2);
- `bt2` filtering statistics by source and GlotLID label.

## Error handling

- GlotLID fails to download or `fasttext` fails to import → a clear error with instructions.
- `backtranslate` with a sample or block incompatible with what was already generated → error.
- Interrupted training → `--resume`.

## Tests (pytest, no real models)

- GlotLID filter with a mocked classifier: accepted labels, threshold, Spanish and digit rules.
- Threshold calibration: picks the value that lets through ≥ 95 % of a list of probabilities.
- Loading Wikipedia from a toy parquet.
- `translate(sample=True)` passes `do_sample=True`, `top_k=10`, `num_beams=1` to the generator (mocked model).
- `--bt-tag` affects only back-translated examples; `--upsample-parallel` repeats exactly N times.
- `DATA_FILES` resolves `bt2`. `backtranslate --name bt2` writes to `data/train/bt2/` without touching `bt.jsonl`. An incompatible sample or block → error.

## Definition of done

1. `bt2` generated with 500,000 sentences and `stats.json`.
2. E3 and E4 trained and evaluated, with rows in `results/SUMMARY.md`.
3. Conclusions: thresholds for E3 and E4, effect of the tag, effect of scaling and filtering, and next step.

## Out of scope

Cloud, full fine-tuning, manual sources or web crawling, human evaluation, and the minor leftovers from subproject 1 that do not affect this round.
