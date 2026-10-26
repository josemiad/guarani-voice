#!/usr/bin/env bash
# Uso: scripts/piper_train.sh <name> <data> espeak|text [--resume] [args extra de LightningCLI]
# Entrena Piper *medium* DESDE CERO (sin --ckpt_path: las voces publicadas heredan licencias de investigación).
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; data=$(realpath -e "data/train/tts/$2"); kind=$3; shift 3
resume=""
if [ "${1:-}" = "--resume" ]; then shift; resume=1; fi
run=runs/$name
if [ -z "$resume" ]; then rm -rf "$run"; fi
mkdir -p "$run"; run=$(realpath "$run")
case $kind in
  espeak) ptype=(); csv=metadata.csv ;;
  text) ptype=(--data.phoneme_type text); csv=metadata_text.csv ;;  # texto ya en símbolos de Piper (tts.piper_text)
  *) echo "tipo de entrada no válido: $kind (espeak|text)" >&2; exit 2 ;;
esac
ckpt=()
if [ -n "$resume" ]; then
  last=$(ls "$run"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | tail -1)
  [ -n "$last" ] && ckpt=(--ckpt_path "$last")
fi
exec envs/piper/.venv/bin/python -m piper.train fit \
  --data.voice_name "$name" \
  --data.csv_path "$data/$csv" \
  --data.audio_dir "$data/wavs" \
  --data.espeak_voice gn \
  "${ptype[@]}" \
  --data.cache_dir "$run/cache" \
  --data.config_path "$run/config.json" \
  --data.validation_split 0.03 \
  --data.batch_size 32 \
  --data.num_workers 4 \
  --model.sample_rate 22050 \
  --trainer.default_root_dir "$run" \
  --trainer.max_time 00:22:00:00 \
  "${ckpt[@]}" "$@"
