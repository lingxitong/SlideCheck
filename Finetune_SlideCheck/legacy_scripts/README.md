# Legacy Training Scripts

These scripts have been **replaced** by the unified training script in the new `slidecheck` library.

## Deprecated Scripts

- `train_mixup.py` - Replaced by `tools/train_phase1.py --variant mixup`
- `train_mixup_feat_perturb.py` - Replaced by `tools/train_phase1.py --variant mixup_fp`
- `train_input_concat_v2.py` - Replaced by `tools/train_phase1.py --variant concatv2`

## Migration Guide

### Old Usage (Deprecated)
```bash
# Old way - using separate scripts
python Finetune_SlideCheck/train_mixup.py --pt_path data.pt --epochs 200
python Finetune_SlideCheck/train_mixup_feat_perturb.py --pt_path data.pt --epochs 200
python Finetune_SlideCheck/train_input_concat_v2.py --pt_path data.pt --epochs 200
```

### New Usage (Recommended)
```bash
# New way - unified script with variants
python tools/train_phase1.py --variant mixup --pt_path data.pt --epochs 200
python tools/train_phase1.py --variant mixup_fp --pt_path data.pt --epochs 200
python tools/train_phase1.py --variant concatv2 --pt_path data.pt --epochs 200

# Or use YAML config
python experiments/scripts/run_phase1.py --config experiments/configs/phase1_mixup_fp.yaml
```

## Benefits of New Approach

1. **Single unified script** - No need to maintain multiple copies
2. **Consistent interface** - All variants use the same parameters
3. **Better organized** - Part of the `slidecheck` library
4. **YAML configs** - Reproducible experiments with configuration files
5. **More variants** - Supports 7 variants: baseline, mixup, feat_perturb, mixup_fp, concatv2, concatv2_mixup, concatv2_mixup_fp

## Why Keep These Files?

These files are kept for reference only. They will be removed in a future release.

**Do not use these scripts for new experiments.**
