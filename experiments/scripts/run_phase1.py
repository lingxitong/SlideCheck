#!/usr/bin/env python3
"""
Run Phase 1 training from YAML configuration

Usage:
    python experiments/scripts/run_phase1.py --config experiments/configs/phase1_mixup_fp.yaml
    python experiments/scripts/run_phase1.py --config experiments/configs/phase1_mixup.yaml
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def build_command(config):
    """Build command from config"""
    cmd = [
        sys.executable,
        'tools/train_phase1.py',
        '--variant', config['variant'],
        '--feat_dir', config['data']['feat_dir'],
        '--test_ratio', str(config['data']['test_ratio']),
        '--foundation_model', config['foundation_model'],
        '--hidden_dim', str(config['model']['hidden_dim']),
        '--dropout', str(config['model']['dropout']),
        '--epochs', str(config['training']['epochs']),
        '--batch_size', str(config['training']['batch_size']),
        '--lr', str(config['training']['lr']),
        '--weight_decay', str(config['training']['weight_decay']),
        '--lambda_constraint', str(config['training']['lambda_constraint']),
        '--early_stop_patience', str(config['training']['early_stop_patience']),
        '--seed', str(config['training']['seed']),
        '--log_dir', config['output']['log_dir'],
    ]

    # Add experiment name if specified
    if 'experiment_name' in config:
        cmd.extend(['--exp_name', config['experiment_name']])

    # Add augmentation parameters if present
    if 'augmentation' in config:
        if 'mixup_alpha' in config['augmentation']:
            cmd.extend(['--mixup_alpha', str(config['augmentation']['mixup_alpha'])])
        if 'perturb_sigma' in config['augmentation']:
            cmd.extend(['--perturb_sigma', str(config['augmentation']['perturb_sigma'])])

    return cmd


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 training from config')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--dry_run', action='store_true', help='Print command without running')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"Phase 1 Training - {config.get('experiment_name', 'Unnamed')}")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Variant: {config['variant']}")
    print(f"Foundation Model: {config['foundation_model']}")

    if 'expected_metrics' in config:
        print(f"\nExpected metrics:")
        for k, v in config['expected_metrics'].items():
            print(f"  {k}: {v:.4f}")

    print(f"{'='*60}\n")

    # Build command
    cmd = build_command(config)

    if args.dry_run:
        print("Command (dry run):")
        print(" ".join(cmd))
        return

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
