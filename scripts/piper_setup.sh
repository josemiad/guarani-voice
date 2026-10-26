#!/usr/bin/env bash
# Crea envs/piper: clon fijado de piper1-gpl (GPL-3.0, solo entrenamiento en servidor) + venv + monotonic_align.
set -euo pipefail
cd "$(dirname "$0")/.."
SHA=efffbfb226bfb511ebbcf55d0cecd8b35a89743d
mkdir -p envs/piper
[ -d envs/piper/repo ] || git clone -q https://github.com/OHF-Voice/piper1-gpl envs/piper/repo
git -C envs/piper/repo checkout -q $SHA
[ -d envs/piper/.venv ] || uv venv -q -p 3.12 envs/piper/.venv
uv pip install -q -p envs/piper/.venv -e "envs/piper/repo[train]" "scikit-build<1" ninja "cmake<4" onnxscript
# UTMOS (val_mos de Piper) necesita torchaudio; la última versión publicada es la 2.11 y funciona con torch 2.14.
uv pip install -q -p envs/piper/.venv "torchaudio==2.11.0" --no-deps --index-url https://download.pytorch.org/whl/cu130
(cd envs/piper/repo && source ../.venv/bin/activate && ./build_monotonic_align.sh && python setup.py build_ext --inplace -q)
envs/piper/.venv/bin/python -c "import torch, piper.train; from piper import espeakbridge; print('piper ok', torch.__version__, torch.cuda.is_available())"
