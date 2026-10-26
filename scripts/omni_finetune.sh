#!/usr/bin/env bash
# Uso: scripts/omni_finetune.sh <name> <model_card> [data_name]
# Afina Omnilingual con la receta oficial de fairseq2 y deja el run en runs/<name>.
# MAX_ELEMENTS (muestras de audio por lote) y GRAD_ACCUM ajustan la memoria: 1_920_000 = 120 s cabe en 24 GB con 300M.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; card=$2; data=$(realpath "data/train/omni/${3:-$name}")
env=envs/omnilingual
# Commit con el que se entrenaron S1 y S2b (receta compatible con fairseq2 0.6).
RECIPE_SHA=81f51e224ce9e74b02cc2a3eaf21b2d91d743455
[ -d $env/repo ] || git clone -q https://github.com/facebookresearch/omnilingual-asr $env/repo
git -C $env/repo checkout -q $RECIPE_SHA
mkdir -p runs/$name/assets
run=$(realpath runs/$name); assets=$run/assets
cat > "$assets/guanani_cv17.yaml" <<EOF
name: guanani_cv17
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: $data/version=0
tokenizer_ref: omniASR_tokenizer_written_v2
EOF
sed -e "s|@MODEL@|$card|" -e "s|@DATA@|$data|" \
  -e "s|max_num_elements: .*|max_num_elements: ${MAX_ELEMENTS:-1_920_000}|" \
  -e "s|num_batches: .*|num_batches: ${GRAD_ACCUM:-4}|" $env/ctc-finetune.yaml > "$run/config.yaml"
cd $env/repo
FAIRSEQ2_USER_ASSET_DIR="$assets" ../.venv/bin/python -m workflows.recipes.wav2vec2.asr \
  "$run" --config-file "$run/config.yaml"
