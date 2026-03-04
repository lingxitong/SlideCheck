"""
Checkpoint utilities - unified format with multi-FM support

Standard format:
{
    'state_dict': model.state_dict(),
    'config': {
        'arch': 'baseline',
        'in_dim': 2560,
        'foundation_model': 'virchow2',  # Records which FM was used
        'hidden_dim': 768,
        'dropout': 0.1
    },
    'epoch': int,
    'metrics': {...}
}

Compatible with legacy formats: model_state_dict, slidecheck_state_dict
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any


# Priority order for finding state_dict in checkpoint
CKPT_KEY_PRIORITY = ['state_dict', 'model_state_dict', 'slidecheck_state_dict']


def load_checkpoint(
    ckpt_path: str,
    foundation_model: Optional[str] = None,
    map_location: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load checkpoint with automatic format adaptation

    Args:
        ckpt_path: Path to checkpoint file
        foundation_model: Current FM name (for dimension validation)
        map_location: Device to load to

    Returns:
        Dict with keys: state_dict, config, epoch, metrics

    Raises:
        KeyError: If no state_dict found in checkpoint
        ValueError: If checkpoint FM dimension doesn't match current FM
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Find state_dict with priority order
    state_dict = None
    for key in CKPT_KEY_PRIORITY:
        if key in ckpt:
            state_dict = ckpt[key]
            break

    if state_dict is None:
        raise KeyError(
            f"Checkpoint {ckpt_path} does not contain state_dict. "
            f"Available keys: {list(ckpt.keys())}"
        )

    config = ckpt.get('config', {})

    # Validate Foundation Model compatibility
    if foundation_model:
        from ..foundation_models import get_foundation_model

        fm_adapter = get_foundation_model(foundation_model)
        fm_config = fm_adapter.get_config()
        expected_dim = fm_config.feature_dim

        ckpt_in_dim = config.get('in_dim')
        ckpt_fm = config.get('foundation_model')

        if ckpt_in_dim and ckpt_in_dim != expected_dim:
            raise ValueError(
                f"Checkpoint in_dim={ckpt_in_dim} does not match "
                f"current FM '{foundation_model}' feature_dim={expected_dim}.\n"
                f"Checkpoint was trained with FM: {ckpt_fm or 'unknown'}"
            )

    return {
        'state_dict': state_dict,
        'config': config,
        'epoch': ckpt.get('epoch', 0),
        'metrics': ckpt.get('metrics', {})
    }


def save_checkpoint(
    model,
    save_path: str,
    foundation_model: str,
    epoch: int = 0,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None
):
    """
    Save checkpoint in unified format

    Args:
        model: Model to save
        save_path: Path to save checkpoint
        foundation_model: FM name (required)
        epoch: Current epoch
        metrics: Training metrics
        config: Additional config
    """
    from ..foundation_models import get_foundation_model

    fm_adapter = get_foundation_model(foundation_model)
    fm_config = fm_adapter.get_config()

    ckpt = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'config': {
            'foundation_model': foundation_model,
            'in_dim': fm_config.feature_dim,
            **(config or {})
        }
    }

    if metrics:
        ckpt['metrics'] = metrics

    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to {save_path}")
