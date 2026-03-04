# CoEvolution_SlideCheck - Legacy Directory

This directory contains the **original experimental scripts** for co-evolution training. These scripts have been **refactored and integrated** into the new `slidecheck` library.

## Status: DEPRECATED ⚠️

**Do not use these scripts for new experiments.** Use the new unified scripts in `tools/` instead.

## Migration Guide

### Step 0: Generate Manifest
**Old**: `CoEvolution_SlideCheck/Step0_Generate_Manifest/generate_manifest.py`
**New**: Use your own manifest generation or adapt the script

### Step 1: MIL Training
**Old**: `CoEvolution_SlideCheck/Step1_MIL_Mining/train_mil_gated.py`
**New**: `tools/train_mil.py`

```bash
# Old way
python CoEvolution_SlideCheck/Step1_MIL_Mining/train_mil_gated.py --manifest train.csv --feature_dir /path/to/features

# New way
python tools/train_mil.py --manifest train.csv --feature_dir /path/to/features --model_type gated --hidden_dim 256
```

### Step 2: Mining Inference
**Old**: `CoEvolution_SlideCheck/Step2_Mining_Inference/mining_inference.py`
**New**: `tools/mining_inference.py`

```bash
# Old way
python CoEvolution_SlideCheck/Step2_Mining_Inference/mining_inference.py --mil_ckpt model.pt --manifest wsi.csv

# New way
python tools/mining_inference.py --mil_ckpt model.pt --manifest wsi.csv --feature_dir /path/to/features --top_k 100
```

### Step 3: Pseudo-Label Generation
**Old**: Multiple scripts in `Step3_Pseudo_Labels/`
**New**: `tools/generate_pseudo_labels.py`

```bash
# Old way (multiple steps)
python CoEvolution_SlideCheck/Step3_Pseudo_Labels/extract_mining_signals.py ...
python CoEvolution_SlideCheck/Step3_Pseudo_Labels/generate_pseudo_labels.py ...
python CoEvolution_SlideCheck/Step3_Pseudo_Labels/build_training_cache.py ...

# New way (single unified script)
python tools/generate_pseudo_labels.py \
  --slidecheck_ckpt phase1_model.pt \
  --mining_cache training_cache.pt \
  --cancer_threshold 0.7 \
  --abnormal_threshold 0.7 \
  --use_confidence_weighting \
  --output pseudo_labels.pt
```

### Step 4: Co-Evolution Training
**Old**: `train_coevolution_v4a.py`, `train_coevolution_v6.py` (separate scripts)
**New**: `tools/train_coevolution.py` (unified script)

```bash
# Old way - V4a
python CoEvolution_SlideCheck/Step4_CoEvolution_Train/train_coevolution_v4a.py --mode ft --cache_path cache.pt

# Old way - V6
python CoEvolution_SlideCheck/Step4_CoEvolution_Train/train_coevolution_v6.py --mode scratch --cache_path cache.pt

# New way - Unified script
python tools/train_coevolution.py --route v4a --mode ft --pseudo_labels pseudo_labels.pt
python tools/train_coevolution.py --route v6 --mode scratch --pseudo_labels pseudo_labels.pt

# Or use YAML config
python experiments/scripts/run_coevolution.py --config experiments/configs/coevolution_v4a.yaml
```

### Step 5: Evaluation
**Old**: `CoEvolution_SlideCheck/Step5_Evaluation/evaluate.py`
**New**: `tools/evaluate_bracs.py`

```bash
# Old way
python CoEvolution_SlideCheck/Step5_Evaluation/evaluate.py --models "Model:ckpt.pt:mlp_v1"

# New way
python tools/evaluate_bracs.py --models "Model:ckpt.pt:baseline" --bracs_path /path/to/BRACS.pt
```

## Complete Pipeline

**Old way** (manual steps):
```bash
cd CoEvolution_SlideCheck
python Step1_MIL_Mining/train_mil_gated.py ...
python Step2_Mining_Inference/mining_inference.py ...
python Step3_Pseudo_Labels/generate_pseudo_labels.py ...
python Step4_CoEvolution_Train/train_coevolution_v4a.py ...
python Step5_Evaluation/evaluate.py ...
```

**New way** (one command):
```bash
bash tools/pipeline/run_coevolution_pipeline.sh --config experiments/configs/coevolution_v4a.yaml
```

## Key Improvements

1. **Unified Interface**: All scripts use consistent parameter names
2. **Single Script per Task**: No need to run multiple scripts for one step
3. **YAML Configs**: Reproducible experiments with configuration files
4. **Better Error Handling**: Clear error messages and validation
5. **Foundation Model Support**: Automatic dimension adaptation for different FMs
6. **Checkpoint Compatibility**: Unified checkpoint format with backward compatibility
7. **Complete Pipeline**: One-command execution of entire workflow

## Directory Structure

```
CoEvolution_SlideCheck/  (DEPRECATED)
├── Step0_Generate_Manifest/
├── Step1_MIL_Mining/
├── Step2_Mining_Inference/
├── Step3_Pseudo_Labels/
├── Step4_CoEvolution_Train/
├── Step5_Evaluation/
├── datasets/  (replaced by slidecheck/datasets/)
└── models/    (replaced by slidecheck/models/)
```

## Recommendation

**Archive this directory** after verifying that the new scripts work correctly:

```bash
# Create archive
tar -czf CoEvolution_SlideCheck_legacy.tar.gz CoEvolution_SlideCheck/

# Move to archive location
mv CoEvolution_SlideCheck_legacy.tar.gz experiments/results/legacy/

# Remove directory (after verification)
# rm -rf CoEvolution_SlideCheck/
```

## Questions?

See the main README or documentation in `experiments/README.md` for more details.

---

**Last Updated**: 2026-03-03
**Status**: Deprecated - Use new `slidecheck` library instead
