#!/usr/bin/env python3
"""
Unified inference script for SlideCheck

Usage:
    python tools/infer.py --ckpt model.pt --input features.h5 --output results.json
    python tools/infer.py --ckpt model.pt --input_dir features/ --output_dir results/
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from slidecheck.inference import SlideCheckPredictor


def main():
    parser = argparse.ArgumentParser(
        description='SlideCheck unified inference script'
    )

    # Model arguments
    parser.add_argument(
        '--ckpt', required=True,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--foundation_model', default=None,
        help='Foundation model name (optional, read from checkpoint)'
    )

    # Input arguments
    parser.add_argument(
        '--input', default=None,
        help='Path to single HDF5 file'
    )
    parser.add_argument(
        '--input_dir', default=None,
        help='Directory containing HDF5 files'
    )
    parser.add_argument(
        '--h5_key', default='features',
        help='Key in HDF5 files (default: features)'
    )

    # Output arguments
    parser.add_argument(
        '--output', default=None,
        help='Output JSON file for single input'
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory for batch processing'
    )

    # Inference arguments
    parser.add_argument(
        '--device', default='cuda:0',
        help='Device to use (default: cuda:0)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=512,
        help='Batch size for inference (default: 512)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input is None and args.input_dir is None:
        parser.error('Either --input or --input_dir must be specified')

    if args.input and args.input_dir:
        parser.error('Cannot specify both --input and --input_dir')

    if args.input and not args.output:
        parser.error('--output must be specified when using --input')

    if args.input_dir and not args.output_dir:
        parser.error('--output_dir must be specified when using --input_dir')

    # Initialize predictor
    print(f"Initializing SlideCheck predictor...")
    predictor = SlideCheckPredictor(
        args.ckpt,
        device=args.device,
        foundation_model=args.foundation_model
    )

    # Single file inference
    if args.input:
        print(f"\nProcessing {args.input}...")
        results = predictor.predict_from_h5(
            args.input,
            h5_key=args.h5_key,
            batch_size=args.batch_size
        )

        # Convert to serializable format
        output_data = {
            'input': args.input,
            'num_patches': len(results['prob_abn']),
            'mean_prob_abn': float(results['prob_abn'].mean()),
            'mean_prob_can': float(results['prob_can'].mean()),
            'max_prob_abn': float(results['prob_abn'].max()),
            'max_prob_can': float(results['prob_can'].max()),
        }

        # Save results
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to {args.output}")
        print(f"  Mean prob_abn: {output_data['mean_prob_abn']:.4f}")
        print(f"  Mean prob_can: {output_data['mean_prob_can']:.4f}")

    # Batch inference
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all HDF5 files
        h5_files = list(input_dir.glob('*.h5')) + list(input_dir.glob('*.hdf5'))

        if not h5_files:
            print(f"No HDF5 files found in {input_dir}")
            return

        print(f"\nProcessing {len(h5_files)} files...")

        for h5_path in tqdm(h5_files):
            try:
                results = predictor.predict_from_h5(
                    str(h5_path),
                    h5_key=args.h5_key,
                    batch_size=args.batch_size
                )

                # Save results
                output_data = {
                    'input': str(h5_path),
                    'num_patches': len(results['prob_abn']),
                    'mean_prob_abn': float(results['prob_abn'].mean()),
                    'mean_prob_can': float(results['prob_can'].mean()),
                    'max_prob_abn': float(results['prob_abn'].max()),
                    'max_prob_can': float(results['prob_can'].max()),
                }

                output_path = output_dir / f"{h5_path.stem}.json"
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)

            except Exception as e:
                print(f"Error processing {h5_path}: {e}")

        print(f"\nBatch inference complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
