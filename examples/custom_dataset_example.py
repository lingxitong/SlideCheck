#!/usr/bin/env python3
"""
Example: Using SlideCheck with your own dataset

This example shows how to:
1. Use the SlideCheck library with custom data
2. Build and train a SlideCheck model
3. Run inference on new data
"""

import sys
from pathlib import Path

# Add SlideCheck to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slidecheck.models import build_slidecheck_model
from slidecheck.inference import SlideCheckPredictor
from slidecheck.utils import save_checkpoint, load_checkpoint


def example_build_model():
    """Example: Build a SlideCheck model"""
    print("=" * 60)
    print("Example 1: Building SlideCheck models")
    print("=" * 60)

    # Build baseline model with Virchow2 (2560d features)
    model_v2 = build_slidecheck_model(
        arch='baseline',
        foundation_model='virchow2',
        hidden_dim=768,
        dropout=0.1
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model_v2.parameters()):,}")

    # Build ConcatV2 model
    model_concat = build_slidecheck_model(
        arch='concatv2',
        foundation_model='virchow2',
        hidden_dim=1024,
        dropout=0.1
    )
    print(f"ConcatV2 parameters: {sum(p.numel() for p in model_concat.parameters()):,}")


def example_inference():
    """Example: Run inference with SlideCheck"""
    print("\n" + "=" * 60)
    print("Example 2: Running inference")
    print("=" * 60)

    # Note: This requires an actual checkpoint file
    # Uncomment and modify the path to run
    """
    predictor = SlideCheckPredictor(
        ckpt_path='path/to/your/checkpoint.pt',
        device='cuda:0'
    )

    # Predict from HDF5 file
    results = predictor.predict_from_h5('path/to/features.h5')

    print(f"Number of patches: {len(results['prob_abn'])}")
    print(f"Mean abnormal probability: {results['prob_abn'].mean():.4f}")
    print(f"Mean cancer probability: {results['prob_can'].mean():.4f}")
    """

    print("\nTo run inference, uncomment the code above and provide:")
    print("  1. Path to a trained checkpoint")
    print("  2. Path to an HDF5 file with features")


def example_checkpoint_handling():
    """Example: Save and load checkpoints"""
    print("\n" + "=" * 60)
    print("Example 3: Checkpoint handling")
    print("=" * 60)

    # Build a model
    model = build_slidecheck_model(
        arch='baseline',
        foundation_model='virchow2',
        hidden_dim=768
    )

    # Save checkpoint (example - not actually saving)
    print("\nTo save a checkpoint:")
    print("""
    from slidecheck.utils import save_checkpoint

    save_checkpoint(
        model=model,
        save_path='my_model.pt',
        foundation_model='virchow2',
        epoch=10,
        metrics={'val_acc': 0.85},
        config={'arch': 'baseline', 'hidden_dim': 768}
    )
    """)

    # Load checkpoint (example)
    print("\nTo load a checkpoint:")
    print("""
    from slidecheck.utils import load_checkpoint

    ckpt = load_checkpoint(
        ckpt_path='my_model.pt',
        foundation_model='virchow2'  # Validates dimension compatibility
    )

    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded model from epoch {ckpt['epoch']}")
    """)


def example_custom_dataset():
    """Example: Adapting SlideCheck to your dataset"""
    print("\n" + "=" * 60)
    print("Example 4: Using your own dataset")
    print("=" * 60)

    print("""
To use SlideCheck with your own data:

1. Extract features using your foundation model:
   - Virchow2: 2560d features
   - UNI: 1024d features (coming soon)
   - GigaPath: 1536d features (coming soon)

2. Save features in HDF5 format:
   import h5py
   with h5py.File('slide001.h5', 'w') as f:
       f.create_dataset('features', data=your_features)

3. Build and train a model:
   model = build_slidecheck_model(
       arch='baseline',
       foundation_model='virchow2'  # Match your feature extractor
   )

4. Run inference:
   predictor = SlideCheckPredictor('trained_model.pt')
   results = predictor.predict_from_h5('slide001.h5')

For training, see experiments/scripts/ for complete examples.
    """)


if __name__ == '__main__':
    example_build_model()
    example_inference()
    example_checkpoint_handling()
    example_custom_dataset()

    print("\n" + "=" * 60)
    print("For more examples, see:")
    print("  - experiments/scripts/ for training examples")
    print("  - tools/ for command-line tools")
    print("=" * 60)
