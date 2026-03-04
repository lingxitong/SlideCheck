# SlideCheck Experiments

This directory contains complete experimental configurations and scripts for reproducing the SlideCheck paper results.

## Directory Structure

```
experiments/
├── configs/          # YAML configuration files for experiments
│   ├── phase1_mixup_fp.yaml      # Best Phase 1 config
│   ├── phase1_mixup.yaml         # Second best Phase 1 config
│   └── phase1_baseline.yaml      # Baseline config
├── scripts/          # Experiment execution scripts
│   └── run_phase1.py             # Phase 1 training runner
├── data/            # Data manifests and configurations (gitignored)
└── results/         # Experiment results (gitignored)
```

## Quick Start

### 1. Phase 1 Training (Base Model)

Train the base SlideCheck model with Mixup + Feature Perturbation (best configuration):

```bash
# Edit config file to set your data paths
vim experiments/configs/phase1_mixup_fp.yaml

# Run training
python experiments/scripts/run_phase1.py --config experiments/configs/phase1_mixup_fp.yaml
```

**Alternative configurations:**

```bash
# Baseline + Mixup (second best)
python experiments/scripts/run_phase1.py --config experiments/configs/phase1_mixup.yaml

# Pure baseline (no augmentation)
python experiments/scripts/run_phase1.py --config experiments/configs/phase1_baseline.yaml
```

### 2. Evaluate on BRACS

Evaluate trained models on BRACS dataset:

```bash
python tools/evaluate_bracs.py \
  --ckpt logs/phase1/phase1_mixup_fp_best/best_model.pt \
  --bracs_path /path/to/BRACS.pt \
  --output results/bracs_eval.json
```

### 3. Direct Training (Without Config File)

You can also train directly using command-line arguments:

```bash
python tools/train_phase1.py \
  --variant mixup_fp \
  --feat_dir /path/to/features \
  --foundation_model virchow2 \
  --hidden_dim 768 \
  --epochs 200 \
  --batch_size 2048 \
  --lr 0.001 \
  --mixup_alpha 0.4 \
  --perturb_sigma 0.1 \
  --log_dir ./logs/phase1 \
  --exp_name my_experiment
```


## Available Training Variants

The unified training script supports multiple variants:

| Variant | Description | Expected Performance |
|---------|-------------|---------------------|
| `baseline` | Pure baseline (no augmentation) | Cancer AUC: 0.90, Abnormal AUC: 0.74 |
| `mixup` | Baseline + Mixup (α=0.4) | Cancer AUC: 0.93, Abnormal AUC: 0.90 |
| `feat_perturb` | Baseline + Feature Perturbation | Cancer AUC: 0.88, Abnormal AUC: 0.79 |
| `mixup_fp` | **Baseline + Mixup + FP (BEST)** | **Cancer AUC: 0.93, Abnormal AUC: 0.90** |
| `concatv2` | ConcatV2 architecture (raw+zscore) | Cancer AUC: 0.91, Abnormal AUC: 0.85 |
| `concatv2_mixup` | ConcatV2 + Mixup | Cancer AUC: 0.91, Abnormal AUC: 0.85 |
| `concatv2_mixup_fp` | ConcatV2 + Mixup + FP | Cancer AUC: 0.92, Abnormal AUC: 0.85 |

**Recommendation**: Use `mixup_fp` for best performance.

## Configuration Files

Configuration files use YAML format and specify:
- Data paths
- Model architecture
- Training hyperparameters
- Augmentation settings
- Expected metrics (for validation)

Example configuration structure:

```yaml
experiment_name: "phase1_mixup_fp_best"
variant: "mixup_fp"
foundation_model: "virchow2"

data:
  feat_dir: "/path/to/features"
  test_ratio: 0.1

model:
  arch: "baseline"
  hidden_dim: 768
  dropout: 0.1

training:
  epochs: 200
  batch_size: 2048
  lr: 0.001
  weight_decay: 0.0001
  lambda_constraint: 1.0
  early_stop_patience: 30
  seed: 2024

augmentation:
  mixup_alpha: 0.4
  perturb_sigma: 0.1

output:
  log_dir: "./logs/phase1"

expected_metrics:
  cancer_auc: 0.9346
  abnormal_auc: 0.9012
```

## Adapting to Your Dataset

To use these experiments with your own data:

1. **Extract features** using Dataset_Preprocess scripts
   ```bash
   python Dataset_Preprocess/SlideCheck_Dataset_Preprocess.py \
     --image_dir /path/to/images \
     --output_dir /path/to/features \
     --foundation_model virchow2
   ```

2. **Update data paths** in config files
   - Edit `experiments/configs/phase1_mixup_fp.yaml`
   - Set `data.feat_dir` to your feature directory

3. **Run training**
   ```bash
   python experiments/scripts/run_phase1.py \
     --config experiments/configs/phase1_mixup_fp.yaml
   ```

4. **Evaluate on your test set**
   ```bash
   python tools/evaluate_bracs.py \
     --ckpt logs/phase1/best_model.pt \
     --bracs_path /path/to/your_test_data.pt
   ```

See `examples/custom_dataset_example.py` for detailed guidance.

## Expected Results

Each configuration file includes expected metrics for validation:

### Phase 1 Results (on BRACS dataset)

| Configuration | Cancer AUC | Cancer BACC | Abnormal AUC | Abnormal BACC | Mean BACC |
|--------------|-----------|-------------|-------------|---------------|-----------|
| **Mixup + FP** | **0.9346** | **0.8676** | 0.9012 | 0.8212 | **0.8444** |
| Mixup | 0.9297 | 0.8568 | **0.9022** | **0.8343** | 0.8456 |
| Baseline | 0.9028 | 0.8400 | 0.7394 | 0.6800 | 0.7600 |

**Key findings:**
- Mixup (α=0.4) is the most critical augmentation (+22% Abnormal AUC)
- Feature Perturbation (σ=0.1) provides additional boost when combined with Mixup
- Simple architecture (2-layer MLP, h=768) + augmentation = best performance

## Training Data

The experiments use the following training data:

**Version A (859,379 samples)**:
- Path: `/path/to/features_virchow2/`
- 8 datasets: CRC-100K, ESCA_TCGA, RenalCell, Spider_Breast, Spider_Colorectal, Spider_Skin, Spider_Thorax, TCGA_Uniform
- Features: 2560d (Virchow2 CLS + MeanPatch)
- Labels: Binary (normal/abnormal, non-cancer/cancer)

**BRACS Test Data**:
- Path: `/path/to/BRACS_virchow2.pt`
- 4,539 ROI samples
- 7 classes: N(0), PB(1), UDH(2), FEA(3), ADH(4), DCIS(5), IC(6)
- Cancer: DCIS(5) + IC(6) vs others
- Abnormal: PB-IC(1-6) vs N(0)

## Output Structure

Training outputs are saved to the log directory:

```
logs/phase1/
└── phase1_mixup_fp_best/
    ├── config.json           # Training configuration
    ├── best_model.pt         # Best checkpoint (by mean BACC)
    ├── best_metrics.json     # Best validation metrics
    ├── final_model.pt        # Final checkpoint
    ├── final_metrics.json    # Final validation metrics
    └── history.json          # Training history
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (e.g., 2048 → 1024)
- Use smaller model (`hidden_dim: 512`)

### Poor Performance
- Check data paths are correct
- Verify feature dimensions match foundation model
- Try different random seeds
- Increase training epochs

### Slow Training
- Increase `batch_size` if GPU memory allows
- Reduce `num_workers` in dataloader
- Use mixed precision training (coming soon)

## Notes

- All experiments use Virchow2 (2560d) features by default
- Training scripts automatically validate checkpoint compatibility
- Results are saved to `experiments/results/` (gitignored)
- Configuration files are version-controlled for reproducibility
- Early stopping patience: 30 epochs (can be adjusted)
- Best checkpoint is selected by mean balanced accuracy

## Co-Evolution Training (Coming Soon)

Phase 2 and Phase 3 training scripts are planned:
- MIL-based mining
- Pseudo-label generation
- Co-evolution training (V4a, V6 routes)

Stay tuned for updates!
