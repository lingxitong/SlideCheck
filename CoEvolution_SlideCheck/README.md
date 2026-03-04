# CoEvolution SlideCheck - 协同训练 Pipeline

基于伪标签的 WSI (Whole Slide Image) 协同训练 pipeline，利用弱监督的 WSI-level 标注生成 patch-level 伪标签，增训 SlideCheck 模型以提升跨域泛化能力。

## Pipeline 流程

```
Phase1 基础模型 (Dataset A patch-level 训练)
         |
         v
Step 0: 生成 Bag 清单 (WSI H5 -> manifest CSV)
         |
         v
Step 1: MIL Mining 训练 (Gated ABMIL + SlideCheck guided)
         |
         v
Step 2: Mining 推理 (产出 per-patch cancer_prob)
         |
         v
Step 3: 伪标签生成 + 训练缓存构建
         |
         v
Step 4: 协同增训 (Finetune / Scratch)
         |
         v
Step 5: 跨域评测
```

## 目录结构

```
CoEvolution_SlideCheck/
├── models/                          # 模型定义
│   ├── slidecheck_mlp.py            # SlideCheckMLP (Baseline, 2560d -> dual heads)
│   ├── gated_abmil_2560.py          # Gated Attention-Based MIL
│   └── abmil_2560.py                # Standard ABMIL
├── datasets/                        # 数据集
│   ├── h5_bag_dataset.py            # WSI Bag Dataset (从 H5 加载 patch features)
│   └── patch_dataset.py             # Patch Dataset (Dataset A .pt 文件)
├── Step0_Generate_Manifest/         # 生成 Bag 清单
├── Step1_MIL_Mining/                # MIL 训练
├── Step2_Mining_Inference/          # Mining 推理
├── Step3_Pseudo_Labels/             # 伪标签生成 + 缓存构建
├── Step4_CoEvolution_Train/         # 协同增训
└── Step5_Evaluation/                # 跨域评测
```

## 数据要求

### Dataset A (patch-level 标注)
- 格式: `.pt` 文件，包含 `embeddings`, `cancer_noncancer_labels`, `normal_abnormal_labels`
- 特征: Virchow2 提取的 2560d embedding

### Dataset B (WSI-level 标注)
- 格式: H5 文件，每个 WSI 一个文件，包含 `features` (N_patches x 2560)
- 清单: CSV 文件，字段 `h5_path`, `label` (0/1)

## 使用方法

### 前提: Phase1 基础模型
使用 `Finetune_SlideCheck/` 下的脚本训练基础模型:

```bash
# 方案 1: Mixup + Feature Perturbation (推荐, Cancer AUC 0.9346)
python Finetune_SlideCheck/train_mixup_feat_perturb.py \
  --feat_dir /path/to/dataset_a_features --epochs 200

# 方案 2: 纯 Mixup (Abnormal BACC 最优 0.8343)
python Finetune_SlideCheck/train_mixup.py \
  --feat_dir /path/to/dataset_a_features --epochs 200

# 方案 3: ConcatV2 (raw+zscore 5120d, 内部验证最高)
python Finetune_SlideCheck/train_input_concat_v2.py \
  --feat_dir /path/to/dataset_a_features --epochs 200
```

### Step 0: 生成 Bag 清单

```bash
python CoEvolution_SlideCheck/Step0_Generate_Manifest/generate_manifest.py \
  --trident_root /path/to/wsi_h5_features \
  --output_dir ./manifests
```

产出: `bag_manifest_train.csv`, `bag_manifest_test.csv`

### Step 1: MIL Mining 训练

```bash
python CoEvolution_SlideCheck/Step1_MIL_Mining/train_mil_gated.py \
  --train_manifest manifests/bag_manifest_train.csv \
  --test_manifest manifests/bag_manifest_test.csv \
  --slidecheck_ckpt phase1_best_model.pt \
  --log_dir ./logs/mil
```

产出: `logs/mil/best_mil_model.pt`

### Step 2: Mining 推理

```bash
python CoEvolution_SlideCheck/Step2_Mining_Inference/mining_inference.py \
  --manifest manifests/bag_manifest_all.csv \
  --slidecheck_ckpt phase1_best_model.pt \
  --mil_ckpt logs/mil/best_mil_model.pt \
  --output mining_inference.pt
```

产出: `mining_inference.pt` (每个 WSI 的 per-patch cancer_prob + attention scores)

### Step 3: 伪标签生成 + 缓存构建

```bash
# 3a. 生成伪标签
python CoEvolution_SlideCheck/Step3_Pseudo_Labels/generate_pseudo_labels.py \
  --inference_results mining_inference.pt \
  --output pseudo_labels.pt

# 3b. 构建训练缓存 (PL patches + Dataset A patches)
python CoEvolution_SlideCheck/Step3_Pseudo_Labels/build_training_cache.py \
  --pseudo_labels pseudo_labels.pt \
  --manifest manifests/bag_manifest_train.csv \
  --dataset_a_dir /path/to/dataset_a_features \
  --output training_cache.pt

# 3c. 提取 mining 信号 (用于 V3c 置信度加权 + 分层采样)
python CoEvolution_SlideCheck/Step3_Pseudo_Labels/extract_mining_signals.py \
  --mining_results mining_inference.pt \
  --pseudo_labels pseudo_labels.pt \
  --manifest manifests/bag_manifest_train.csv \
  --dataset_a_dir /path/to/dataset_a_features \
  --output mining_signals.pt
```

### Step 4: 协同增训

提供两条路线:

#### 路线 A: V4a (综合最优, Mean BACC 0.8180)
特点: 负袋数据增强 + Type2 cancer_weight=0 + V3c 动态权重

```bash
# Finetune 模式 (5 epochs, lr=1e-5)
python CoEvolution_SlideCheck/Step4_CoEvolution_Train/train_coevolution_v4a.py \
  --mode ft \
  --cache_path training_cache.pt \
  --neg_bag_cache neg_bag_cache.pt \
  --signals_path mining_signals.pt \
  --slidecheck_ckpt phase1_best_model.pt \
  --log_dir ./logs

# Scratch 模式 (200 epochs, lr=1e-3, early stop)
python CoEvolution_SlideCheck/Step4_CoEvolution_Train/train_coevolution_v4a.py \
  --mode scratch \
  --cache_path training_cache.pt \
  --neg_bag_cache neg_bag_cache.pt \
  --signals_path mining_signals.pt \
  --log_dir ./logs
```

#### 路线 B: V6 (V3c 置信度加权 + 分层采样)
特点: 更精细的样本加权 + 域平衡采样

```bash
# V6c (随机采样, Finetune 最优 0.8090)
python CoEvolution_SlideCheck/Step4_CoEvolution_Train/train_coevolution_v6.py \
  --mode finetune --no_stratified \
  --cache_path training_cache.pt \
  --signals_path mining_signals.pt \
  --slidecheck_ckpt phase1_best_model.pt \
  --log_dir ./logs

# V5a (无分层, Scratch 最优 0.8158)
python CoEvolution_SlideCheck/Step4_CoEvolution_Train/train_coevolution_v6.py \
  --mode scratch --no_stratified \
  --cache_path training_cache.pt \
  --signals_path mining_signals.pt \
  --log_dir ./logs

# V6b (sqrt 分层采样, 温和版)
python CoEvolution_SlideCheck/Step4_CoEvolution_Train/train_coevolution_v6.py \
  --mode scratch --balance_mode sqrt \
  --cache_path training_cache.pt \
  --signals_path mining_signals.pt \
  --log_dir ./logs
```

### Step 5: 评测

```bash
# 评测单个模型
python CoEvolution_SlideCheck/Step5_Evaluation/evaluate.py \
  --models "V4a BL FT:logs/v4a_mixup/final.pt:baseline" \
  --bracs_path /path/to/BRACS.pt \
  --unitopatho_path /path/to/UNITOPATHO.pt \
  --camel_path /path/to/CAMEL.pt \
  --output_csv results.csv

# 批量评测 (通过模型列表文件)
python CoEvolution_SlideCheck/Step5_Evaluation/evaluate.py \
  --model_list models.txt \
  --bracs_path /path/to/BRACS.pt \
  --output_csv results.csv
```

models.txt 格式:
```
# name:checkpoint_path:arch
Phase1 BL:checkpoints/phase1_best.pt:baseline
V4a BL FT:logs/v4a/final.pt:baseline
V6c BL FT:logs/v6c/final.pt:baseline
```

## Checkpoint 格式

所有脚本保存的 checkpoint 均包含 `state_dict` key，与 `Infer_SlideCheck/` 推理脚本兼容:
```python
# 加载推理
ckpt = torch.load('model.pt', map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
```

## 最优模型实验结果

在 BRACS + UNITOPATHO + CAMEL 三个外部数据集上的 Mean BACC:

| 排名 | 模型 | Mean BACC | 脚本 | 参数 |
|------|------|-----------|------|------|
| 1 | V4a BL FT | 0.8180 | train_coevolution_v4a.py | --mode ft |
| 2 | V5a BL SCR | 0.8158 | train_coevolution_v6.py | --mode scratch --no_stratified |
| 3 | V6c BL FT | 0.8090 | train_coevolution_v6.py | --mode finetune --no_stratified |
