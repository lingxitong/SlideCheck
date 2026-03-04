#!/usr/bin/env python3
"""
训练缓存构建 — 从伪标签 + Dataset A 构建训练缓存

输入:
  - pseudo_labels.pt: 伪标签文件 (由 generate_pseudo_labels.py 产出)
  - bag_manifest_train.csv: WSI Bag 清单
  - Dataset A 特征目录: 含 .pt 文件的目录

输出:
  - training_cache.pt: 包含 pl_emb/pl_can/pl_abn + da_emb/da_can/da_abn
"""
import argparse, csv, os, sys, time
from pathlib import Path
import h5py, torch, numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="构建训练缓存")
    p.add_argument("--pseudo_labels", type=str, required=True,
                   help="伪标签文件路径 (pseudo_labels.pt)")
    p.add_argument("--manifest", type=str, required=True,
                   help="WSI Bag 清单 CSV (bag_manifest_train.csv)")
    p.add_argument("--dataset_a_dir", type=str, required=True,
                   help="Dataset A 特征目录 (含 .pt 文件)")
    p.add_argument("--output", type=str, required=True,
                   help="输出缓存文件路径 (training_cache.pt)")
    return p.parse_args()


args = parse_args()
PL_PATH = args.pseudo_labels
MANIFEST = args.manifest
DA_PATH = args.dataset_a_dir
OUT_PATH = args.output

t0 = time.time()
sys.stdout.reconfigure(line_buffering=True)

# 1. Build manifest dict
manifest = {}
with open(MANIFEST) as f:
    for row in csv.DictReader(f):
        manifest[Path(row["h5_path"]).stem] = row["h5_path"]
print(f"Manifest: {len(manifest)} entries")

# 2. Load V5a pseudo label embeddings
pl_data = torch.load(PL_PATH, map_location="cpu", weights_only=False)
pseudo_labels = pl_data["pseudo_labels"]
pl_stats = pl_data.get("stats", {})
print(f"Pseudo labels V5a: {len(pseudo_labels)} WSIs")
if pl_stats:
    total_pl = pl_stats.get('n_labeled_patches', '?')
    print(f"  Expected labeled patches: {total_pl}")

all_emb, all_can, all_abn = [], [], []
missing = 0
loaded = 0
for i, (wsi_id, info) in enumerate(pseudo_labels.items()):
    if wsi_id not in manifest:
        missing += 1
        continue

    indices = info["indices"]
    cancer_labels = info["cancer_labels"]
    abnormal_labels = info["abnormal_labels"]
    repeat = int(info.get("repeat", 1))

    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.long)
    if not isinstance(cancer_labels, torch.Tensor):
        cancer_labels = torch.tensor(cancer_labels, dtype=torch.float32)
    if not isinstance(abnormal_labels, torch.Tensor):
        abnormal_labels = torch.tensor(abnormal_labels, dtype=torch.float32)

    h5_path = manifest[wsi_id]
    with h5py.File(h5_path, "r") as hf:
        feats = torch.from_numpy(hf["features"][:]).float()
    patch_feats = feats[indices]
    del feats

    for _ in range(repeat):
        all_emb.append(patch_feats)
        all_can.append(cancer_labels)
        all_abn.append(abnormal_labels)

    loaded += 1
    if (loaded) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  [{loaded}/{len(pseudo_labels)}] {elapsed:.0f}s elapsed")

pl_emb = torch.cat(all_emb, dim=0)
pl_can = torch.cat(all_can, dim=0)
pl_abn = torch.cat(all_abn, dim=0)
print(f"Pseudo labels done: {pl_emb.shape[0]} patches ({pl_emb.shape[1]}d), {missing} missing WSIs")
print(f"  Time: {time.time()-t0:.1f}s")

# 统计
t1 = ((pl_can == 1) & (pl_abn == 1)).sum().item()
t2 = ((pl_can == 0) & (pl_abn == 1)).sum().item()
t3 = ((pl_can == 0) & (pl_abn == 0)).sum().item()
print(f"  Type 1 (cancer+abnormal): {t1}")
print(f"  Type 2 (noncancer+abnormal): {t2}")
print(f"  Type 3 (normal): {t3}")

# 3. Load Dataset A
da_emb_list, da_can_list, da_abn_list = [], [], []
for f in sorted(os.listdir(DA_PATH)):
    if not f.endswith('.pt'):
        continue
    d = torch.load(os.path.join(DA_PATH, f), map_location='cpu', weights_only=False)
    da_emb_list.append(torch.as_tensor(d['embeddings'], dtype=torch.float32))
    da_can_list.append(torch.as_tensor(d['cancer_noncancer_labels'], dtype=torch.float32))
    da_abn_list.append(torch.as_tensor(d['normal_abnormal_labels'], dtype=torch.float32))
    print(f"  Dataset A: {f} -> {d['embeddings'].shape[0]} samples")

da_emb = torch.cat(da_emb_list, dim=0)
da_can = torch.cat(da_can_list, dim=0)
da_abn = torch.cat(da_abn_list, dim=0)
print(f"Dataset A done: {da_emb.shape[0]} patches")

# 4. 总计
total = pl_emb.shape[0] + da_emb.shape[0]
total_cancer = (pl_can == 1).sum().item() + (da_can == 1).sum().item()
total_abn = (pl_abn == 1).sum().item() + (da_abn == 1).sum().item()
print(f"\n{'='*50}")
print(f"V5a Training Data Summary:")
print(f"  Pseudo labels: {pl_emb.shape[0]} patches")
print(f"  Dataset A:     {da_emb.shape[0]} patches")
print(f"  Total:         {total} patches")
print(f"  Cancer=1:      {total_cancer} ({total_cancer/total*100:.1f}%)")
print(f"  Abnormal=1:    {total_abn} ({total_abn/total*100:.1f}%)")
print(f"  Cancer=0:      {total - total_cancer} ({(total-total_cancer)/total*100:.1f}%)")
print(f"{'='*50}")

# 5. Save cache
cache = {
    "pl_emb": pl_emb, "pl_can": pl_can, "pl_abn": pl_abn,
    "da_emb": da_emb, "da_can": da_can, "da_abn": da_abn,
}
torch.save(cache, OUT_PATH)
sz = os.path.getsize(OUT_PATH) / (1024**3)
print(f"\nSaved cache: {OUT_PATH} ({sz:.2f} GB)")
print(f"Total time: {time.time()-t0:.1f}s")
