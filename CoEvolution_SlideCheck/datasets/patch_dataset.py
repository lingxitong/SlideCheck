"""
Patch Dataset A — 用于 Phase 3 SlideCheck 进修时的防遗忘

读取 Version A 的 85 万 patch 数据 (8 个 pt 文件)。
Phase 3 Step 3.2 中, 伪标签 patches + Dataset A patches 混合训练。

pt 文件格式:
    {
        'embeddings': [N, 2560],
        'cancer_noncancer_labels': [N],   # 0=Non-Cancer, 1=Cancer
        'normal_abnormal_labels': [N],    # 0=Normal, 1=Abnormal
        'img_names': list,
    }
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class PatchDatasetA(Dataset):
    """
    Dataset A 的 Patch 数据 (强监督, 有 Patch 标签)

    支持:
      1. 传入目录 → 自动加载目录下所有 pt 文件
      2. 传入单个 pt 文件路径列表
    """

    def __init__(
        self,
        data_source: str,
        label_type: str = 'cancer',
    ):
        """
        Args:
            data_source: pt 文件目录 或 单个 pt 文件路径
            label_type: 'cancer' 使用 cancer_noncancer_labels,
                        'abnormal' 使用 normal_abnormal_labels
        """
        self.label_type = label_type

        # 收集 pt 文件
        source = Path(data_source)
        if source.is_dir():
            pt_files = sorted(source.glob('*.pt'))
        else:
            pt_files = [source]

        # 加载
        all_embeddings = []
        all_labels = []

        for pt_file in pt_files:
            data = torch.load(str(pt_file), map_location='cpu', weights_only=False)
            emb = data['embeddings']
            if emb.dim() == 3:
                emb = emb.squeeze(0)
            all_embeddings.append(torch.as_tensor(emb, dtype=torch.float32))

            if label_type == 'cancer':
                lab = data['cancer_noncancer_labels']
            else:
                lab = data['normal_abnormal_labels']
            all_labels.append(torch.as_tensor(lab, dtype=torch.long))

        self.embeddings = torch.cat(all_embeddings, dim=0)
        self.labels = torch.cat(all_labels, dim=0)
        self.n_samples = len(self.embeddings)

        print(f"PatchDatasetA loaded: {self.n_samples} samples from {len(pt_files)} files")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'embedding': self.embeddings[idx],   # [2560]
            'label': self.labels[idx],            # scalar
        }


def create_patch_dataloader_a(
    data_source: str,
    batch_size: int = 2048,
    shuffle: bool = True,
    num_workers: int = 4,
    label_type: str = 'cancer',
) -> DataLoader:
    """创建 Dataset A 的 Patch DataLoader"""
    dataset = PatchDatasetA(data_source, label_type=label_type)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
