#!/usr/bin/env python3
"""
提取 mining 信号，与训练缓存对齐。

输入:
  - mining_inference.pt: 每个 WSI 的 per-patch cancer_prob (由 mining_inference.py 产出)
  - pseudo_labels.pt: 伪标签文件
  - bag_manifest_train.csv: WSI Bag 清单
  - Dataset A 特征目录

输出:
  - mining_signals.pt: {pl_cancer_prob, da_sizes, da_names}

注意: 必须与 build_training_cache.py 按相同顺序遍历 WSI，以确保 cancer_prob 对齐。
"""
import argparse, csv, os, sys, time
from pathlib import Path
import torch

sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    p = argparse.ArgumentParser(description="提取 mining 信号")
    p.add_argument("--mining_results", type=str, required=True,
                   help="Mining 推理结果文件 (mining_inference.pt)")
    p.add_argument("--pseudo_labels", type=str, required=True,
                   help="伪标签文件 (pseudo_labels.pt)")
    p.add_argument("--manifest", type=str, required=True,
                   help="WSI Bag 清单 CSV")
    p.add_argument("--dataset_a_dir", type=str, required=True,
                   help="Dataset A 特征目录")
    p.add_argument("--output", type=str, required=True,
                   help="输出文件路径 (mining_signals.pt)")
    return p.parse_args()


args = parse_args()
MINING_PATH = args.mining_results
PL_PATH = args.pseudo_labels
MANIFEST = args.manifest
DA_PATH = args.dataset_a_dir
OUT_PATH = args.output

t0 = time.time()

# 1. Build manifest dict (same logic as build_training_cache_v5a.py)
manifest = {}
with open(MANIFEST) as f:
    for row in csv.DictReader(f):
        manifest[Path(row["h5_path"]).stem] = row["h5_path"]
print(f"Manifest: {len(manifest)} entries")

# 2. Load mining inference results
print("Loading mining inference results...")
mining = torch.load(MINING_PATH, map_location="cpu", weights_only=False)
print(f"Mining results: {len(mining)} WSIs")

# 3. Load V5a pseudo labels
print("Loading pseudo labels V5a...")
pl_data = torch.load(PL_PATH, map_location="cpu", weights_only=False)
pseudo_labels = pl_data["pseudo_labels"]
print(f"Pseudo labels: {len(pseudo_labels)} WSIs")

# 4. Extract cancer_prob in SAME ORDER as build_training_cache_v5a.py
all_cancer_prob = []
missing = 0
loaded = 0

for wsi_id, info in pseudo_labels.items():
    # Same skip logic as build_training_cache_v5a.py
    if wsi_id not in manifest:
        missing += 1
        continue

    indices = info["indices"]
    repeat = int(info.get("repeat", 1))

    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.long)

    # Get cancer_prob from mining results
    wsi_mining = mining[wsi_id]
    cp = wsi_mining["cancer_prob"]
    if not isinstance(cp, torch.Tensor):
        cp = torch.tensor(cp)
    cp = cp[indices].float()

    for _ in range(repeat):
        all_cancer_prob.append(cp)

    loaded += 1

pl_cancer_prob = torch.cat(all_cancer_prob)
print(f"Extracted cancer_prob: {pl_cancer_prob.shape[0]} patches ({missing} WSIs missing)")
print(f"  mean={pl_cancer_prob.mean():.4f}, min={pl_cancer_prob.min():.4f}, max={pl_cancer_prob.max():.4f}")

# 5. Compute Dataset A file sizes (for stratified sampling)
da_sizes = []
da_names = []
for f in sorted(os.listdir(DA_PATH)):
    if not f.endswith('.pt'):
        continue
    d = torch.load(os.path.join(DA_PATH, f), map_location='cpu', weights_only=False)
    n = d['embeddings'].shape[0]
    da_sizes.append(n)
    da_names.append(f.replace('.pt', ''))
    print(f"  DA: {f} -> {n} samples")
    del d

print(f"Dataset A: {len(da_sizes)} files, {sum(da_sizes)} total samples")

# 6. Save
output = {
    "pl_cancer_prob": pl_cancer_prob,
    "da_sizes": da_sizes,
    "da_names": da_names,
}
torch.save(output, OUT_PATH)
sz = os.path.getsize(OUT_PATH) / (1024**2)
print(f"\nSaved: {OUT_PATH} ({sz:.1f} MB)")
print(f"Total time: {time.time()-t0:.1f}s")
