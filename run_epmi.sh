#!/bin/bash
#SBATCH --job-name=epmi_diag
#SBATCH --account=bexq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/u/yxu30/moe-epmi/epmi_%j.out
#SBATCH --error=/u/yxu30/moe-epmi/epmi_%j.err

# ── HuggingFace caches ────────────────────────────────────────────────────────
export HF_HOME=/work/hdd/bexq/yxu30/hf
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers   # where OLMoE weights are cached
export HF_DATASETS_CACHE=$HF_HOME/datasets

# ── HF auth ───────────────────────────────────────────────────────────────────
if [ -f "$HOME/.secrets/hf_token" ]; then
  export HUGGINGFACE_HUB_TOKEN="$(cat $HOME/.secrets/hf_token)"
fi

PYTHON=/projects/bexq/yxu30/conda/envs/olmoe/bin/python
WORKDIR=/u/yxu30/moe-epmi

echo "[job] started on $(hostname) at $(date)"
echo "[job] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "[job] Python: $($PYTHON --version)"

cd "$WORKDIR"
$PYTHON run_diagnostics.py

echo "[job] finished at $(date)"
