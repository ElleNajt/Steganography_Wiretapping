#!/usr/bin/env bash
set -euo pipefail

REPO_ID="lnajt/steganography-wiretapping"
DB_PATH="results/stego.db"

if [ ! -f "$DB_PATH" ]; then
    echo "Error: $DB_PATH not found"
    exit 1
fi

echo "Uploading $DB_PATH ($(du -h "$DB_PATH" | cut -f1)) to $REPO_ID..."
hf upload "$REPO_ID" "$DB_PATH" "stego.db" \
    --repo-type dataset \
    --commit-message "Update stego.db $(date +%Y-%m-%d)"

echo "Done: https://huggingface.co/datasets/$REPO_ID"
