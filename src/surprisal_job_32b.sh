#!/bin/bash
#SBATCH --job-name=surprisal_32b
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]
#SBATCH --output=/workspace-vast/lnajt/logs/surprisal_32b_%j.out

cd /workspace-vast/lnajt/Steganography_Wiretapping

python3 -m venv --system-site-packages .venv 2>/dev/null
source .venv/bin/activate
pip install transformers torch accelerate --quiet

python3 -u src/surprisal_decoder.py \
    --model Qwen/Qwen2.5-32B \
    --input data/qa_trials_for_surprisal.json \
    --output results/surprisal_results_32b.json
