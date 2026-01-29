#!/bin/bash
# Train UWSAM model on UIIS10K using SAM ViT-B + LoRA
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/train_uiis10k_vitb.sh

set -e

cd "$(dirname "$0")/.."

CONFIG="project/our/configs/multiclass_uiis10k_vitb_lora.py"

echo "=== Training UWSAM on UIIS10K with ViT-B + LoRA ==="
echo "Config: $CONFIG"
echo ""

# Single GPU training
python tools/train.py "$CONFIG"
