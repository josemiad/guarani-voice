"""Exporta un checkpoint de Piper a ONNX con el exportador clásico (se ejecuta en envs/piper).

torch ≥ 2.9 usa por defecto el exportador dynamo, que falla con el VITS de Piper; forzamos dynamo=False.
Uso: envs/piper/.venv/bin/python scripts/piper_export.py --checkpoint X.ckpt --output-file Y.onnx
"""

import functools
import runpy

import torch

torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
runpy.run_module("piper.train.export_onnx", run_name="__main__")
