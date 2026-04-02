#!/bin/bash
#SBATCH --job-name=surprisal_gemini
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22],node-28
#SBATCH --output=/workspace-vast/lnajt/logs/surprisal_gemini_%j.out

cd /workspace-vast/lnajt/steganography
source .venv/bin/activate

python3 -u src/surprisal_decoder.py \
    --model Qwen/Qwen2.5-7B \
    --input data/qa_trials_gemini_for_surprisal.json \
    --output results/surprisal_results_gemini_7b.json
