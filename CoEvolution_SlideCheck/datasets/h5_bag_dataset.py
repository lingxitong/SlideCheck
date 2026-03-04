"""
H5 Bag Dataset — 读取 TRIDENT 生成的 h5 特征文件

TRIDENT h5 格式 (每个 WSI 一个文件):
    {
        'features': [N_patches, 2560],   # float32, Virchow2 特征
        'coords':   [N_patches, 2],      # int64, patch 坐标
    }

标签通过 manifest CSV 提供 (由 generate_manifest.py 生成):
    h5_path,bag_label,dataset,category
    /abs/path/to/xxx.h5,0,esophagus,non-tumor
    /abs/path/to/yyy.h5,1,esophagus,squa
    ...

设计决策:
  - 懒加载 (lazy loading): __init__ 只读 manifest, __getitem__ 时才读 h5
  - 避免一次性把几百个 WSI 全部加载到内存
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import torch
from torch.utils.data import Dataset, DataLoader


def load_manifest(manifest_path: str) -> List[Dict]:
    """
    读取 manifest CSV, 返回记录列表。

    每条记录: {'h5_path': str, 'bag_label': int, 'dataset': str, 'category': str}
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
    基于 manifest 的 Bag 数据集, 懒加载 h5 文件。

    用法:
        dataset = H5BagDataset('bag_manifest.csv')
        bag = dataset[0]
        # bag['features']:  [N, 2560] tensor
        # bag['label']:     scalar tensor (0 或 1)
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
            manifest_path: manifest CSV 路径
            label_filter: 只保留指定标签的 bag (如 [0, 1] 排除 -1)
        """
        self.manifest_path = manifest_path
        self.records = load_manifest(manifest_path)

        # 过滤
        if label_filter is not None:
            self.records = [r for r in self.records if r['bag_label'] in label_filter]

        # 验证文件存在 (只检查, 不加载)
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

        # 懒加载 h5
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
        """数据集统计 (不加载 h5)"""
        labels = [r['bag_label'] for r in self.records]
        datasets = [r['dataset'] for r in self.records]

        stats = {
            'n_bags': len(self.records),
            'n_label_0': sum(1 for l in labels if l == 0),
            'n_label_1': sum(1 for l in labels if l == 1),
        }

        # 按数据集统计
        from collections import Counter
        ds_counts = Counter(datasets)
        for ds, cnt in ds_counts.items():
            stats[f'dataset_{ds}'] = cnt

        return stats


def bag_collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Bag 级 collate: 每个 bag 的 patch 数量不同, 不能 stack。
    直接返回 list。
    """
    return batch


def create_bag_dataloader(
    manifest_path: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 2,
    label_filter: Optional[List[int]] = None,
) -> DataLoader:
    """创建 Bag DataLoader"""
    dataset = H5BagDataset(manifest_path, label_filter=label_filter)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=bag_collate_fn,
        pin_memory=False,  # h5 懒加载, pin_memory 没意义
    )
