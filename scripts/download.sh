#!/usr/bin/env bash
# Descarga los conjuntos de evaluación públicos a data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/jojajovai data/flores data/common_voice

base=https://raw.githubusercontent.com/pln-fing-udelar/jojajovai/main/data
for f in jojajovai_all.csv jojajovai_sample_annotations.csv; do
  curl -fsSL -o "data/jojajovai/$f" "$base/$f"
done
echo "Jojajovai OK"

mkdir -p data/americasnlp data/kuatia data/agustin
anlp=https://raw.githubusercontent.com/AmericasNLP/americasnlp2021/main/data/guarani-spanish
for f in train.gn train.es; do
  curl -fsSL -o "data/americasnlp/$f" "$anlp/$f"
done
echo "AmericasNLP OK"

uv run python - <<'EOF' || echo "FLORES+ falló: acepta las condiciones en https://huggingface.co/datasets/openlanguagedata/flores_plus y ejecuta 'uv run hf auth login'"
import shutil
from huggingface_hub import hf_hub_download
for split in ("dev", "devtest"):
    for code in ("gug_Latn", "spa_Latn"):
        p = hf_hub_download("openlanguagedata/flores_plus", f"{split}/{code}.jsonl", repo_type="dataset")
        shutil.copy(p, f"data/flores/{split}.{code}.jsonl")
print("FLORES+ OK")
EOF

uv run python - <<'EOF' || echo "Common Voice falló: acepta las condiciones en https://huggingface.co/datasets/fsicoli/common_voice_17_0"
# Réplica en HF de Common Voice 17.0 (CC0); Mozilla ya no publica Common Voice en HF.
import shutil, tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download
repo = "fsicoli/common_voice_17_0"
d = Path("data/common_voice"); (d / "clips").mkdir(parents=True, exist_ok=True)
from huggingface_hub import HfApi
files = HfApi().list_repo_files(repo, repo_type="dataset")
for split in ("test", "dev", "train", "other"):
    shutil.copy(hf_hub_download(repo, f"transcript/gn/{split}.tsv", repo_type="dataset"), d / f"{split}.tsv")
    for tar in sorted(f for f in files if f.startswith(f"audio/gn/{split}/") and f.endswith(".tar")):
        with tarfile.open(hf_hub_download(repo, tar, repo_type="dataset")) as t:
            for m in t.getmembers():
                if m.isfile():
                    m.name = Path(m.name).name
                    t.extract(m, d / "clips", filter="data")
print("Common Voice OK")
EOF

uv run python - <<'EOF' || echo "KuatIA falló: acepta las condiciones en https://huggingface.co/datasets/guaran-ia/kuatia"
import shutil
from huggingface_hub import hf_hub_download
from guanani.train.corpus import KUATIA_SUBSETS
for s in KUATIA_SUBSETS:
    p = hf_hub_download("guaran-ia/kuatia", f"individual_full/{s}_full.jsonl", repo_type="dataset")
    shutil.copy(p, f"data/kuatia/{s}_full.jsonl")
print("KuatIA OK")
EOF

uv run python - <<'EOF' || echo "agustin (opcional) falló: acepta las condiciones en https://huggingface.co/datasets/guaran-ia/agustin-guarani-llm-splits"
import shutil
from huggingface_hub import hf_hub_download
shutil.copy(hf_hub_download("guaran-ia/agustin-guarani-llm-splits", "train.parquet", repo_type="dataset"), "data/agustin/train.parquet")
print("agustin OK")
EOF

uv run python - <<'EOF' || echo "Wikipedia gn falló"
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
Path("data/wikipedia").mkdir(parents=True, exist_ok=True)
p = hf_hub_download("wikimedia/wikipedia", "20231101.gn/train-00000-of-00001.parquet", repo_type="dataset")
shutil.copy(p, "data/wikipedia/gn.parquet")
print("Wikipedia gn OK")
EOF
