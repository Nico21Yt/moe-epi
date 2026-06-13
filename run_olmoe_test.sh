#!/bin/bash
#SBATCH --job-name=olmoe_test
#SBATCH --account=bexq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/u/yxu30/moe-epmi/olmoe_test_%j.out
#SBATCH --error=/u/yxu30/moe-epmi/olmoe_test_%j.err

# HuggingFace cache on large HDD allocation
export HF_HOME=/work/hdd/bexq/yxu30/hf
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

# Load HF token if available
if [ -f "$HOME/.secrets/hf_token" ]; then
  export HUGGINGFACE_HUB_TOKEN="$(cat $HOME/.secrets/hf_token)"
fi

PYTHON=/projects/bexq/yxu30/conda/envs/olmoe/bin/python

echo "[job] started on $(hostname) at $(date)"
echo "[job] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

$PYTHON /u/yxu30/olmoe_expert_probe.py \
    --model allenai/OLMoE-1B-7B-0924 \
    --prompt "Write a short paragraph explaining why mixture-of-experts routing can be efficient."

echo "[job] finished at $(date)"
