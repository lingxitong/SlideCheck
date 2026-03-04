"""
generate_manifest.py — 扫描 TRIDENT_output 目录, 生成 bag_manifest.csv

用法:
  python generate_manifest.py \
    --trident_root /path/to/TRIDENT_output \
    --output_dir /path/to/output

生成:
  bag_manifest_all.csv      — 所有 h5 文件
  bag_manifest_train.csv    — 训练集 (80%)
  bag_manifest_val.csv      — 验证集 (20%)

标签来源:
  食管-非肿瘤-癌前病变-癌:
    - esophagus_wsi_list.csv 中路径前缀 → non-tumor=0, squa=1, ade=1
    - low/high 默认排除 (bag_label=-1)
  食管-腺癌: 全部 Cancer (1)
  低分化癌:  全部 Cancer (1)
"""

import csv
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple


def parse_esophagus_labels(config_json_path: str) -> Dict[str, str]:
    """
    从 TRIDENT _config_segmentation.json 中解析 wsi_rel_paths,
    提取每个 WSI 的类别 (non-tumor/squa/ade/low/high)

    Returns:
        {filename_stem: category}  e.g. {'24-23943-B-HE': 'non-tumor'}
    """
    with open(config_json_path, 'r') as f:
        config = json.load(f)

    mapping = {}
    for rel_path in config.get('wsi_rel_paths', []):
        # rel_path 格式: "non-tumor/24-23943-B-HE.sdpc" 或 "19-01464-N-HE.sdpc"
        parts = rel_path.replace('\\', '/').split('/')
        filename = parts[-1]
        stem = Path(filename).stem  # 去掉扩展名

        if len(parts) >= 2:
            category = parts[0]  # non-tumor, squa, ade, low, high
        else:
            category = 'unknown'

        mapping[stem] = category

    return mapping


CATEGORY_TO_LABEL = {
    'non-tumor': 0,
    'squa': 1,
    'ade': 1,
    'low': -1,   # 默认排除, 可通过参数修改
    'high': -1,  # 默认排除
    'unknown': -1,
}


def scan_dataset(
    dataset_name: str,
    features_dir: str,
    label_func,
) -> List[Dict]:
    """扫描一个数据集的 features_virchow2 目录"""
    features_path = Path(features_dir)
    if not features_path.exists():
        print(f"  [WARN] {features_path} does not exist, skipping")
        return []

    records = []
    for h5_file in sorted(features_path.glob('*.h5')):
        stem = h5_file.stem
        label, category = label_func(stem)

        records.append({
            'h5_path': str(h5_file.resolve()),
            'bag_label': label,
            'dataset': dataset_name,
            'category': category,
        })

    return records


def main(args):
    trident_root = Path(args.trident_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    # === 1. 食管-非肿瘤-癌前病变-癌 ===
    esophagus_dir = trident_root / '食管-非肿瘤-癌前病变-癌'
    config_path = esophagus_dir / '_config_segmentation.json'
    features_dir = esophagus_dir / '20x_224px_0px_overlap' / 'features_virchow2'

    if config_path.exists():
        esophagus_labels = parse_esophagus_labels(str(config_path))
        print(f"食管-主: parsed {len(esophagus_labels)} WSI labels from config")

        def esophagus_label_func(stem):
            cat = esophagus_labels.get(stem, 'unknown')
            label = CATEGORY_TO_LABEL.get(cat, -1)
            return label, cat

        records = scan_dataset('esophagus', str(features_dir), esophagus_label_func)
        all_records.extend(records)
        print(f"  Found {len(records)} h5 files")
    else:
        print(f"  [WARN] {config_path} not found")

    # === 2. 食管-腺癌 (全部 Cancer) ===
    ade_dir = trident_root / '食管-腺癌' / '20x_224px_0px_overlap' / 'features_virchow2'
    records = scan_dataset('esophagus_ade', str(ade_dir), lambda s: (1, 'ade'))
    all_records.extend(records)
    print(f"食管-腺癌: {len(records)} h5 files (all Cancer)")

    # === 3. 低分化癌 (全部 Cancer) ===
    difen_dir = trident_root / '低分化癌' / '20x_224px_0px_overlap' / 'features_virchow2'
    records = scan_dataset('difenhua', str(difen_dir), lambda s: (1, 'difenhua'))
    all_records.extend(records)
    print(f"低分化癌: {len(records)} h5 files (all Cancer)")

    # === 4. 低瘤和高瘤 (默认排除, 可选) ===
    if args.include_diliu:
        diliu_dir = trident_root / '低瘤和高瘤' / '20x_224px_0px_overlap' / 'features_virchow2'
        # 默认: 全部标为 Cancer (高级别上皮内瘤变)
        records = scan_dataset('diliu', str(diliu_dir), lambda s: (1, 'diliu'))
        all_records.extend(records)
        print(f"低瘤高瘤: {len(records)} h5 files (all Cancer)")

    # === 5. MSI (默认排除, 可选) ===
    if args.include_msi:
        msi_dir = trident_root / 'MSI' / '20x_224px_0px_overlap' / 'features_virchow2'
        records = scan_dataset('msi', str(msi_dir), lambda s: (1, 'msi'))
        all_records.extend(records)
        print(f"MSI: {len(records)} h5 files (all Cancer)")

    # === 汇总 ===
    print(f"\n总计: {len(all_records)} records")

    # 过滤有效标签
    valid_records = [r for r in all_records if r['bag_label'] >= 0]
    excluded = len(all_records) - len(valid_records)
    print(f"有效 (label>=0): {len(valid_records)}, 排除: {excluded}")

    n_label0 = sum(1 for r in valid_records if r['bag_label'] == 0)
    n_label1 = sum(1 for r in valid_records if r['bag_label'] == 1)
    print(f"Label 0 (Non-Cancer): {n_label0}")
    print(f"Label 1 (Cancer): {n_label1}")

    # === 写入 all manifest ===
    all_csv = output_dir / 'bag_manifest_all.csv'
    with open(all_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['h5_path', 'bag_label', 'dataset', 'category'])
        writer.writeheader()
        writer.writerows(valid_records)
    print(f"\nSaved: {all_csv}")

    # === Train/Val 分割 ===
    random.seed(args.seed)
    shuffled = valid_records.copy()
    random.shuffle(shuffled)

    n_val = int(len(shuffled) * args.val_ratio)
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]

    train_csv = output_dir / 'bag_manifest_train.csv'
    val_csv = output_dir / 'bag_manifest_val.csv'

    with open(train_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['h5_path', 'bag_label', 'dataset', 'category'])
        writer.writeheader()
        writer.writerows(train_records)

    with open(val_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['h5_path', 'bag_label', 'dataset', 'category'])
        writer.writeheader()
        writer.writerows(val_records)

    n_train_0 = sum(1 for r in train_records if r['bag_label'] == 0)
    n_train_1 = sum(1 for r in train_records if r['bag_label'] == 1)
    n_val_0 = sum(1 for r in val_records if r['bag_label'] == 0)
    n_val_1 = sum(1 for r in val_records if r['bag_label'] == 1)

    print(f"Train: {len(train_records)} (0:{n_train_0}, 1:{n_train_1}) → {train_csv}")
    print(f"Val: {len(val_records)} (0:{n_val_0}, 1:{n_val_1}) → {val_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate bag manifest CSV')
    parser.add_argument('--trident_root', type=str, required=True,
                        help='TRIDENT_output root directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for manifest CSV files')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--include_diliu', action='store_true', default=False,
                        help='Include 低瘤高瘤 dataset')
    parser.add_argument('--include_msi', action='store_true', default=False,
                        help='Include MSI dataset')
    args = parser.parse_args()
    main(args)
