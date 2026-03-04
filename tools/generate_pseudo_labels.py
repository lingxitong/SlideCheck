#!/usr/bin/env python3
"""
Generate Pseudo Labels using SlideCheck model

This script:
1. Loads a trained SlideCheck model
2. Runs inference on mined patches
3. Generates pseudo-labels based on confidence thresholds
4. Saves pseudo-labeled data for co-evolution training

Usage:
    python tools/generate_pseudo_labels.py \
        --slidecheck_ckpt logs/phase1/best_model.pt \
        --mining_cache training_cache.pt \
        --output pseudo_labels.pt \
        --cancer_threshold 0.7 \
        --abnormal_threshold 0.7
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from slidecheck.inference import SlideCheckPredictor


def generate_pseudo_labels(
    predictor,
    features,
    cancer_threshold=0.7,
    abnormal_threshold=0.7,
    use_confidence_weighting=False
):
    """
    Generate pseudo-labels with confidence thresholds

    Args:
        predictor: SlideCheckPredictor instance
        features: Patch features [N, D]
        cancer_threshold: Confidence threshold for cancer
        abnormal_threshold: Confidence threshold for abnormal
        use_confidence_weighting: Use confidence as sample weight

    Returns:
        Dict with pseudo-labels and metadata
    """
    print("\nGenerating pseudo-labels...")

    # Run inference
    results = predictor.predict_from_embeddings(features, batch_size=2048)

    prob_abn = results['prob_abn']
    prob_can = results['prob_can']

    # Generate hard labels
    pseudo_abn = (prob_abn > abnormal_threshold).astype(int)
    pseudo_can = (prob_can > cancer_threshold).astype(int)

    # Compute confidence scores
    conf_abn = np.where(pseudo_abn == 1, prob_abn, 1 - prob_abn)
    conf_can = np.where(pseudo_can == 1, prob_can, 1 - prob_can)

    # Filter by confidence
    high_conf_mask = (conf_abn >= abnormal_threshold) & (conf_can >= cancer_threshold)

    # Compute sample weights if requested
    if use_confidence_weighting:
        weights = (conf_abn + conf_can) / 2
    else:
        weights = np.ones(len(features))

    pseudo_labels = {
        'pseudo_abnormal': pseudo_abn,
        'pseudo_cancer': pseudo_can,
        'prob_abnormal': prob_abn,
        'prob_cancer': prob_can,
        'confidence_abnormal': conf_abn,
        'confidence_cancer': conf_can,
        'high_confidence_mask': high_conf_mask,
        'sample_weights': weights,
        'num_total': len(features),
        'num_high_confidence': high_conf_mask.sum()
    }

    return pseudo_labels


def analyze_pseudo_labels(pseudo_labels, bag_labels=None):
    """
    Analyze pseudo-label distribution

    Args:
        pseudo_labels: Pseudo-labels dict
        bag_labels: Original bag labels (optional)

    Returns:
        Analysis dict
    """
    analysis = {
        'total_patches': pseudo_labels['num_total'],
        'high_confidence_patches': int(pseudo_labels['num_high_confidence']),
        'high_confidence_ratio': float(pseudo_labels['num_high_confidence'] / pseudo_labels['num_total']),
        'pseudo_label_distribution': {
            'abnormal': {
                'negative': int((pseudo_labels['pseudo_abnormal'] == 0).sum()),
                'positive': int((pseudo_labels['pseudo_abnormal'] == 1).sum())
            },
            'cancer': {
                'negative': int((pseudo_labels['pseudo_cancer'] == 0).sum()),
                'positive': int((pseudo_labels['pseudo_cancer'] == 1).sum())
            }
        },
        'confidence_stats': {
            'abnormal': {
                'mean': float(pseudo_labels['confidence_abnormal'].mean()),
                'std': float(pseudo_labels['confidence_abnormal'].std()),
                'min': float(pseudo_labels['confidence_abnormal'].min()),
                'max': float(pseudo_labels['confidence_abnormal'].max())
            },
            'cancer': {
                'mean': float(pseudo_labels['confidence_cancer'].mean()),
                'std': float(pseudo_labels['confidence_cancer'].std()),
                'min': float(pseudo_labels['confidence_cancer'].min()),
                'max': float(pseudo_labels['confidence_cancer'].max())
            }
        }
    }

    # Analyze agreement with bag labels if provided
    if bag_labels is not None:
        bag_abn = np.array([label['abnormal'] for label in bag_labels])
        bag_can = np.array([label['cancer'] for label in bag_labels])

        # Compute agreement rate
        abn_agreement = (pseudo_labels['pseudo_abnormal'] == bag_abn).mean()
        can_agreement = (pseudo_labels['pseudo_cancer'] == bag_can).mean()

        analysis['bag_agreement'] = {
            'abnormal': float(abn_agreement),
            'cancer': float(can_agreement)
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description='Generate Pseudo Labels')

    # Model arguments
    parser.add_argument('--slidecheck_ckpt', required=True,
                        help='Path to trained SlideCheck checkpoint')
    parser.add_argument('--foundation_model', default='virchow2',
                        help='Foundation model name')

    # Data arguments
    parser.add_argument('--mining_cache', required=True,
                        help='Path to mining cache (.pt)')

    # Pseudo-labeling arguments
    parser.add_argument('--cancer_threshold', type=float, default=0.7,
                        help='Confidence threshold for cancer (default: 0.7)')
    parser.add_argument('--abnormal_threshold', type=float, default=0.7,
                        help='Confidence threshold for abnormal (default: 0.7)')
    parser.add_argument('--use_confidence_weighting', action='store_true',
                        help='Use confidence as sample weight')
    parser.add_argument('--filter_low_confidence', action='store_true',
                        help='Filter out low-confidence samples')

    # Output arguments
    parser.add_argument('--output', required=True,
                        help='Output path for pseudo-labels (.pt)')

    # Other arguments
    parser.add_argument('--device', default='cuda:0',
                        help='Device')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Pseudo-Label Generation")
    print(f"{'='*60}")
    print(f"SlideCheck checkpoint: {args.slidecheck_ckpt}")
    print(f"Mining cache: {args.mining_cache}")
    print(f"Cancer threshold: {args.cancer_threshold}")
    print(f"Abnormal threshold: {args.abnormal_threshold}")
    print(f"{'='*60}\n")

    # Load mining cache
    print("Loading mining cache...")
    cache = torch.load(args.mining_cache, map_location='cpu', weights_only=False)

    features = cache['features']
    bag_labels = cache.get('bag_labels', None)

    print(f"Loaded {len(features)} patches from {cache.get('num_bags', 'unknown')} bags\n")

    # Initialize predictor
    print("Loading SlideCheck model...")
    predictor = SlideCheckPredictor(
        args.slidecheck_ckpt,
        device=args.device,
        foundation_model=args.foundation_model
    )
    print()

    # Generate pseudo-labels
    pseudo_labels = generate_pseudo_labels(
        predictor,
        features,
        cancer_threshold=args.cancer_threshold,
        abnormal_threshold=args.abnormal_threshold,
        use_confidence_weighting=args.use_confidence_weighting
    )

    # Analyze pseudo-labels
    print("\nAnalyzing pseudo-labels...")
    analysis = analyze_pseudo_labels(pseudo_labels, bag_labels)

    print(f"\nPseudo-Label Statistics:")
    print(f"  Total patches: {analysis['total_patches']}")
    print(f"  High confidence: {analysis['high_confidence_patches']} "
          f"({analysis['high_confidence_ratio']*100:.1f}%)")
    print(f"\n  Abnormal distribution:")
    print(f"    Negative: {analysis['pseudo_label_distribution']['abnormal']['negative']}")
    print(f"    Positive: {analysis['pseudo_label_distribution']['abnormal']['positive']}")
    print(f"\n  Cancer distribution:")
    print(f"    Negative: {analysis['pseudo_label_distribution']['cancer']['negative']}")
    print(f"    Positive: {analysis['pseudo_label_distribution']['cancer']['positive']}")
    print(f"\n  Confidence (Abnormal):")
    print(f"    Mean: {analysis['confidence_stats']['abnormal']['mean']:.4f}")
    print(f"    Std: {analysis['confidence_stats']['abnormal']['std']:.4f}")
    print(f"\n  Confidence (Cancer):")
    print(f"    Mean: {analysis['confidence_stats']['cancer']['mean']:.4f}")
    print(f"    Std: {analysis['confidence_stats']['cancer']['std']:.4f}")

    if 'bag_agreement' in analysis:
        print(f"\n  Agreement with bag labels:")
        print(f"    Abnormal: {analysis['bag_agreement']['abnormal']*100:.1f}%")
        print(f"    Cancer: {analysis['bag_agreement']['cancer']*100:.1f}%")

    # Filter low-confidence samples if requested
    if args.filter_low_confidence:
        mask = pseudo_labels['high_confidence_mask']
        print(f"\nFiltering low-confidence samples...")
        print(f"  Keeping {mask.sum()} / {len(mask)} patches")

        # Filter all arrays
        for key in ['pseudo_abnormal', 'pseudo_cancer', 'prob_abnormal', 'prob_cancer',
                    'confidence_abnormal', 'confidence_cancer', 'sample_weights']:
            pseudo_labels[key] = pseudo_labels[key][mask]

        features = features[mask]

        # Update cache info
        if 'bag_ids' in cache:
            cache['bag_ids'] = [cache['bag_ids'][i] for i in range(len(mask)) if mask[i]]
        if 'patch_indices' in cache:
            cache['patch_indices'] = [cache['patch_indices'][i] for i in range(len(mask)) if mask[i]]
        if 'attention_scores' in cache:
            cache['attention_scores'] = [cache['attention_scores'][i] for i in range(len(mask)) if mask[i]]

    # Prepare output
    output_data = {
        'features': features,
        'pseudo_abnormal_labels': pseudo_labels['pseudo_abnormal'],
        'pseudo_cancer_labels': pseudo_labels['pseudo_cancer'],
        'prob_abnormal': pseudo_labels['prob_abnormal'],
        'prob_cancer': pseudo_labels['prob_cancer'],
        'confidence_abnormal': pseudo_labels['confidence_abnormal'],
        'confidence_cancer': pseudo_labels['confidence_cancer'],
        'sample_weights': pseudo_labels['sample_weights'],
        'bag_ids': cache.get('bag_ids', []),
        'patch_indices': cache.get('patch_indices', []),
        'attention_scores': cache.get('attention_scores', []),
        'config': {
            'cancer_threshold': args.cancer_threshold,
            'abnormal_threshold': args.abnormal_threshold,
            'use_confidence_weighting': args.use_confidence_weighting,
            'filter_low_confidence': args.filter_low_confidence,
            'slidecheck_ckpt': args.slidecheck_ckpt
        },
        'analysis': analysis
    }

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(output_data, output_path)
    print(f"\nPseudo-labels saved to {output_path}")

    # Save analysis
    analysis_path = output_path.parent / (output_path.stem + '_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis saved to {analysis_path}")

    print(f"\n{'='*60}")
    print("Pseudo-label generation complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
