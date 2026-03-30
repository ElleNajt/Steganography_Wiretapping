#!/bin/bash
#SBATCH --job-name=surprisal_decoder
#SBATCH --partition=general
#SBATCH --qos=low
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --exclude=node-[0-1],node-10,node-12,node-[16-22]
#SBATCH --output=/workspace-vast/lnajt/logs/surprisal_%j.out

cd /workspace-vast/lnajt/Steganography_Wiretapping

pip install transformers torch --quiet

python3 -u src/surprisal_decoder.py \
    --model Qwen/Qwen2.5-0.5B \
    --input data/qa_trials_for_surprisal.json \
    --output results/surprisal_results.json
