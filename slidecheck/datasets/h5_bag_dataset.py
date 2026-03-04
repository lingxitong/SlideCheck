"""
H5 Bag Dataset — Reads TRIDENT-generated h5 feature files

TRIDENT h5 format (one file per WSI):
    {
        'features': [N_patches, 2560],   # float32, Virchow2 features
        'coords':   [N_patches, 2],      # int64, patch coordinates
    }

Labels provided via manifest CSV:
    h5_path,bag_label,dataset,category
    /abs/path/to/xxx.h5,0,esophagus,non-tumor
    /abs/path/to/yyy.h5,1,esophagus,squa
    ...

Design: Lazy loading — __init__ only reads manifest, __getitem__ reads h5 on demand.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import torch
from torch.utils.data import Dataset, DataLoader


def load_manifest(manifest_path: str) -> List[Dict]:
    """
    Read manifest CSV, return list of records.

    Each record: {'h5_path': str, 'bag_label': int, 'dataset': str, 'category': str}
    """
    records = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'h5_path': row['h5_path'],
                'bag_label': int(row['bag_label']),
                'dataset': row.get('dataset', ''),
                'category': row.get('category', ''),
            })
    return records


class H5BagDataset(Dataset):
    """
    Manifest-based Bag dataset with lazy h5 loading.

    Usage:
        dataset = H5BagDataset('bag_manifest.csv')
        bag = dataset[0]
        # bag['features']:  [N, 2560] tensor
        # bag['label']:     scalar tensor (0 or 1)
        # bag['wsi_id']:    str
        # bag['n_patches']: int
    """

    def __init__(
        self,
        manifest_path: str,
        label_filter: Optional[List[int]] = None,
    ):
        """
        Args:
            manifest_path: Path to manifest CSV
            label_filter: Only keep bags with specified labels (e.g. [0, 1] to exclude -1)
        """
        self.manifest_path = manifest_path
        self.records = load_manifest(manifest_path)

        # Filter by label
        if label_filter is not None:
            self.records = [r for r in self.records if r['bag_label'] in label_filter]

        # Validate files exist (check only, don't load)
        missing = [r['h5_path'] for r in self.records if not Path(r['h5_path']).exists()]
        if missing:
            print(f"[WARN] {len(missing)} h5 files not found, will be skipped:")
            for p in missing[:5]:
                print(f"  {p}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
            self.records = [r for r in self.records if Path(r['h5_path']).exists()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        h5_path = record['h5_path']
        wsi_id = Path(h5_path).stem

        # Lazy load h5
        with h5py.File(h5_path, 'r') as f:
            features = torch.from_numpy(f['features'][:]).float()  # [N, 2560]

        return {
            'features': features,
            'label': torch.tensor(record['bag_label'], dtype=torch.long),
            'wsi_id': wsi_id,
            'n_patches': features.shape[0],
            'dataset': record['dataset'],
            'category': record['category'],
        }

    def get_statistics(self) -> Dict:
        """Dataset statistics (without loading h5)"""
        labels = [r['bag_label'] for r in self.records]
        datasets = [r['dataset'] for r in self.records]

        stats = {
            'n_bags': len(self.records),
            'n_label_0': sum(1 for l in labels if l == 0),
            'n_label_1': sum(1 for l in labels if l == 1),
        }

        from collections import Counter
        ds_counts = Counter(datasets)
        for ds, cnt in ds_counts.items():
            stats[f'dataset_{ds}'] = cnt

        return stats


def bag_collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Bag-level collate: each bag has different number of patches, cannot stack.
    Returns list directly.
    """
    return batch


def create_bag_dataloader(
    manifest_path: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 2,
    label_filter: Optional[List[int]] = None,
) -> DataLoader:
    """Create Bag DataLoader"""
    dataset = H5BagDataset(manifest_path, label_filter=label_filter)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=bag_collate_fn,
        pin_memory=False,  # lazy h5 loading, pin_memory is pointless
    )
