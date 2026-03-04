#!/bin/bash
"""
Complete Co-Evolution Pipeline

This script runs the complete co-evolution training pipeline:
1. Train MIL model for mining
2. Run mining inference to extract patches
3. Generate pseudo-labels using SlideCheck
4. Train co-evolution model

Usage:
    bash tools/pipeline/run_coevolution_pipeline.sh \
        --config experiments/configs/coevolution_v4a.yaml
"""

set -e  # Exit on error

# Default values
CONFIG=""
SKIP_MIL=false
SKIP_MINING=false
SKIP_PSEUDO=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --skip-mil)
            SKIP_MIL=true
            shift
            ;;
        --skip-mining)
            SKIP_MINING=true
            shift
            ;;
        --skip-pseudo)
            SKIP_PSEUDO=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    echo "Usage: $0 --config experiments/configs/coevolution_v4a.yaml"
    exit 1
fi

echo "=========================================="
echo "Co-Evolution Training Pipeline"
echo "=========================================="
echo "Config: $CONFIG"
echo "=========================================="
echo ""

# Load config (simplified - in practice would parse YAML)
# For now, assume paths are set as environment variables or in config

# Step 1: Train MIL model (if not skipped)
if [ "$SKIP_MIL" = false ]; then
    echo "Step 1: Training MIL model..."
    python tools/train_mil.py \
        --manifest ${MIL_MANIFEST:-train_manifest.csv} \
        --feature_dir ${FEATURE_DIR:-/path/to/features} \
        --model_type gated \
        --hidden_dim 256 \
        --epochs 50 \
        --output_dir ${MIL_OUTPUT:-./logs/mil}

    MIL_CKPT="${MIL_OUTPUT:-./logs/mil}/best_model.pt"
    echo "MIL model saved to: $MIL_CKPT"
    echo ""
else
    echo "Step 1: Skipping MIL training (using existing model)"
    MIL_CKPT=${MIL_CKPT:-./logs/mil/best_model.pt}
    echo ""
fi

# Step 2: Mining inference (if not skipped)
if [ "$SKIP_MINING" = false ]; then
    echo "Step 2: Running mining inference..."
    python tools/mining_inference.py \
        --mil_ckpt $MIL_CKPT \
        --manifest ${MINING_MANIFEST:-wsi_manifest.csv} \
        --feature_dir ${FEATURE_DIR:-/path/to/features} \
        --top_k 100 \
        --min_attention 0.01 \
        --output ${MINING_OUTPUT:-./data/mining_signals.pt} \
        --build_cache \
        --cache_output ${CACHE_OUTPUT:-./data/training_cache.pt}

    TRAINING_CACHE="${CACHE_OUTPUT:-./data/training_cache.pt}"
    echo "Training cache saved to: $TRAINING_CACHE"
    echo ""
else
    echo "Step 2: Skipping mining inference (using existing cache)"
    TRAINING_CACHE=${TRAINING_CACHE:-./data/training_cache.pt}
    echo ""
fi

# Step 3: Generate pseudo-labels (if not skipped)
if [ "$SKIP_PSEUDO" = false ]; then
    echo "Step 3: Generating pseudo-labels..."
    python tools/generate_pseudo_labels.py \
        --slidecheck_ckpt ${SLIDECHECK_CKPT:-./logs/phase1/best_model.pt} \
        --mining_cache $TRAINING_CACHE \
        --cancer_threshold 0.7 \
        --abnormal_threshold 0.7 \
        --use_confidence_weighting \
        --output ${PSEUDO_OUTPUT:-./data/pseudo_labels.pt}

    PSEUDO_LABELS="${PSEUDO_OUTPUT:-./data/pseudo_labels.pt}"
    echo "Pseudo-labels saved to: $PSEUDO_LABELS"
    echo ""
else
    echo "Step 3: Skipping pseudo-label generation (using existing labels)"
    PSEUDO_LABELS=${PSEUDO_LABELS:-./data/pseudo_labels.pt}
    echo ""
fi

# Step 4: Co-evolution training
echo "Step 4: Running co-evolution training..."
python tools/train_coevolution.py \
    --route ${ROUTE:-v4a} \
    --mode ${MODE:-ft} \
    --slidecheck_ckpt ${SLIDECHECK_CKPT:-./logs/phase1/best_model.pt} \
    --pseudo_labels $PSEUDO_LABELS \
    --epochs ${COEVO_EPOCHS:-5} \
    --lr ${COEVO_LR:-1e-5} \
    --batch_size ${COEVO_BATCH_SIZE:-2048} \
    --type2_cancer_weight ${TYPE2_WEIGHT:-0.0} \
    --use_v3c_weighting \
    --log_dir ${COEVO_LOG_DIR:-./logs/coevolution}

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Results saved to: ${COEVO_LOG_DIR:-./logs/coevolution}"
echo ""
