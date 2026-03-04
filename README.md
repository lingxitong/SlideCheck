# SlideCheck
#### SlideCheck as Prior: Guiding Self-Supervised Pathology Representation Learning with Dataset Distributions

<p align="center">
  <a href='https://arxiv.org/abs/2505.21928'>
  <img src='https://img.shields.io/badge/Arxiv-2505.21928-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://huggingface.co/xtxx/SlideCheck'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow'></a>
  <a href='https://github.com/lingxitong/SlideCheck'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
</p>

<img src="https://github.com/lingxitong/SlideCheck/blob/main/SlideCheck_Logo.png"  width="390px" align="right" />

Self-supervised learning (SSL) has shown strong transferability for pathology foundation models, yet most pipelines still sample patches from whole-slide images (WSIs) uniformly at random despite severe redundancy and imbalanced tissue distributions. We propose SlideCheck as a prior, using supervised distribution priors to guide SSL patch selection. We unify multiple large-scale public ROI datasets and map heterogeneous labels into two binary factors: normal vs. abnormal and cancer vs. non-cancer. With ~1M labeled patches, we train and open-source SlideCheck, a lightweight patch classifier that outputs prior scores for candidate patches. These scores can be used to filter and prioritize diagnostically relevant patches before or during SSL pretraining, reducing uninformative tissue redundancy and improving data efficiency without changing the SSL objective. We hope SlideCheck can serve as a practical, reusable tool to facilitate dataset curation and patch sampling for future pathology SSL research.

SlideCheck is a lightweight Foundation Model Based dual-head patch classifier that outputs two signals for each patch feature:

- **normal vs. abnormal** (`logit_abn`)
- **noncancer vs. cancer** (`logit_can`)

This project was originally developed for our previous work and is continuously maintained to be more user-friendly and support more approaches for histopathology WSI analysis.

**If you find this codebase helpful in your research, please consider citing:**

```bibtex
@article{zhu2025subspecialty,
  title={Subspecialty-specific foundation model for intelligent gastrointestinal pathology},
  author={Zhu, Lianghui and Ling, Xitong and Ouyang, Minxi and Liu, Xiaoping and Guan, Tian and Fu, Mingxi and Cheng, Zhiqiang and Fu, Fanglei and Zeng, Maomao and Liu, Liming and others},
  journal={arXiv preprint arXiv:2505.21928},
  year={2025}
}
```

---

## Repository Structure

```
SlideCheck/
├── slidecheck/                       # Core library
│   ├── models/                       # Model definitions (SlideCheckMLP, GatedABMIL_2560)
│   ├── datasets/                     # Data loading (PatchDataset, H5BagDataset)
│   ├── inference/                    # Inference interface (SlideCheckPredictor)
│   ├── utils/                        # Checkpoint, metrics utilities
│   ├── foundation_models/            # Multi-FM support (Virchow2; UNI, GigaPath planned)
│   ├── training/                     # Training utilities
│   └── mining/                       # Data mining utilities
│
├── tools/                            # Command-line scripts
│   ├── train_phase1.py               # Phase 1: SlideCheck base training
│   ├── train_mil.py                  # Phase 2: Gated Attention MIL training
│   ├── mining_inference.py           # Phase 2: MIL + SlideCheck mining inference
│   ├── generate_pseudo_labels_v5a.py # Phase 3: Pseudo-label generation (V5a)
│   ├── build_v2_cache.py             # Phase 3: Build training cache
│   ├── train_coevolution.py          # Phase 3: Co-evolution training (V4a/V6)
│   ├── evaluate_bracs.py             # Evaluation on BRACS dataset
│   └── infer.py                      # General inference script
│
├── Dataset_Preprocess/               # Feature extraction from raw images
├── Finetune_SlideCheck/              # Legacy finetuning scripts
├── Infer_SlideCheck/                 # Legacy inference scripts
├── CoEvolution_SlideCheck/           # Co-evolution step-by-step guide
│
├── setup.py                          # Package installation
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/lingxitong/SlideCheck.git
cd SlideCheck
pip install -e .
```

### Basic Usage

```python
from slidecheck.inference import SlideCheckPredictor

predictor = SlideCheckPredictor('checkpoint.pt')
results = predictor.predict_from_h5('features.h5')

print(f"Abnormal prob: {results['prob_abn'].mean():.4f}")
print(f"Cancer prob: {results['prob_can'].mean():.4f}")
```

---

## Training Pipeline

SlideCheck uses a 3-phase co-evolution training pipeline:

### Phase 1: Base Training (SlideCheck)

Train the dual-head patch classifier on Dataset A (859K patches, 8 datasets):

```bash
python tools/train_phase1.py \
    --variant mixup_fp \
    --feat_dir /path/to/features_virchow2/ \
    --epochs 200 \
    --batch_size 2048 \
    --lr 1e-3 \
    --seed 2024
```

**Variants**: `baseline`, `mixup`, `feat_perturb`, `mixup_fp` (best), `concatv2`, `concatv2_mixup`, `concatv2_mixup_fp`

**Best result**: Mixup(alpha=0.4) + Feature Perturbation(sigma=0.1) -> BRACS Cancer AUC 0.9346

### Phase 2: MIL Mining

#### Step 2.1: Train Gated Attention MIL

```bash
python tools/train_mil.py \
    --train_manifest bag_manifest.csv \
    --epochs 30
```

#### Step 2.2 (optional): SlideCheck-Guided MIL

```bash
python tools/train_mil.py \
    --train_manifest bag_manifest.csv \
    --epochs 30 \
    --slidecheck_ckpt logs/phase1/best_model.pt \
    --epochs_guided 10
```

#### Step 2.3: Mining Inference

Run both SlideCheck and MIL on all bags to extract per-patch signals:

```bash
python tools/mining_inference.py \
    --manifest bag_manifest.csv \
    --slidecheck_ckpt logs/phase1/best_model.pt \
    --mil_ckpt logs/phase2/step1_best.pt \
    --output mining_inference.pt
```

Output: `{wsi_id: {cancer_prob, abnormal_prob, attention, bag_prob, bag_label}}`

### Phase 3: Co-Evolution Training

#### Step 3.1: Generate Pseudo-Labels (V5a)

```bash
python tools/generate_pseudo_labels_v5a.py \
    --inference_results mining_inference.pt \
    --output pseudo_labels_v5a.pt
```

Three-class intersection strategy:
- **Type 1** (Cancer+Abnormal): cancer_top 5% ∩ attention_top 10%
- **Type 2** (NonCancer+Abnormal): cancer_bottom 20% ∩ abn_top 10% ∩ attn_bottom 20%
- **Type 3** (Normal): all_bottom 5%

#### Step 3.2: Build Training Cache

```bash
python tools/build_v2_cache.py \
    --pseudo_labels pseudo_labels_v5a.pt \
    --h5_manifest bag_manifest.csv \
    --da_dir /path/to/dataset_a/ \
    --output training_cache.pt
```

#### Step 3.3: Co-Evolution Training

**Finetune mode** (from Phase 1 checkpoint):

```bash
python tools/train_coevolution.py \
    --route v4a \
    --mode ft \
    --slidecheck_ckpt logs/phase1/best_model.pt \
    --v2_cache training_cache.pt \
    --neg_bag_cache neg_bag_cache.pt \
    --mining_signals mining_signals.pt
```

**Scratch mode** (from random init):

```bash
python tools/train_coevolution.py \
    --route v4a \
    --mode scratch \
    --v2_cache training_cache.pt \
    --neg_bag_cache neg_bag_cache.pt \
    --mining_signals mining_signals.pt \
    --epochs 200 \
    --lr 1e-3
```

Key V4a features:
- Type 2 cancer weight = 0 (exclude from cancer loss)
- V3c confidence weighting
- Negative bag enhancement
- Mixup(alpha=0.4) + Feature Perturbation(sigma=0.1)

---

## Evaluation

### BRACS Cross-Domain Evaluation

```bash
python tools/evaluate_bracs.py \
    --ckpt logs/phase1/best_model.pt \
    --bracs_path /path/to/BRACS.pt \
    --output results.json
```

---

## Model Architecture

### SlideCheckMLP (Baseline)

```
LayerNorm(2560) -> Linear(2560,768) -> GELU -> Dropout(0.1)
                -> Linear(768,768)   -> GELU -> Dropout(0.1)
                -> head_abn: Linear(768,1)    # Normal/Abnormal
                -> head_can: Linear(768,1)    # Cancer/Non-Cancer
```

Input: Virchow2 features 2560d (CLS 1280d + MeanPool 1280d)

### GatedABMIL_2560 (MIL)

```
attention_a: Linear(2560,384) -> Tanh
attention_b: Linear(2560,384) -> Sigmoid
gated: a * b -> Dropout(0.25) -> Linear(384,1) -> Softmax(dim=0)
bag_feature: weighted sum -> Linear(2560,2)
```

---

## Data Format

### Patch Features (Dataset A)

```python
{
    'embeddings': np.ndarray,              # [N, 2560]
    'normal_abnormal_labels': np.ndarray,  # [N] 0=Normal, 1=Abnormal
    'cancer_noncancer_labels': np.ndarray, # [N] 0=Non-Cancer, 1=Cancer
}
```

### Bag Features (H5, TRIDENT format)

```python
# One h5 file per WSI:
{
    'features': [N_patches, 2560],   # float32
    'coords':   [N_patches, 2],      # int64
}
```

### Bag Manifest CSV

```csv
h5_path,bag_label,dataset,category
/path/to/xxx.h5,0,esophagus,non-tumor
/path/to/yyy.h5,1,esophagus,squa
```

---

## Internal Validation Performance

| Backbone | Task | Acc | BAcc | AUC | AUPRC | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| UNI v1 | Abnormal | 0.9921 | 0.9911 | 0.9996 | 0.9999 | 0.9934 | 0.9888 |
| UNI v1 | Cancer | 0.9928 | 0.9928 | 0.9997 | 0.9997 | 0.9918 | 0.9938 |
| Virchow2 | Abnormal | 0.9933 | 0.9926 | 0.9997 | 0.9999 | 0.9943 | 0.9909 |
| Virchow2 | Cancer | 0.9935 | 0.9935 | 0.9998 | 0.9998 | 0.9940 | 0.9930 |

---

## Legacy Interfaces

Original scripts are preserved for backward compatibility:

```bash
# Legacy inference
python Infer_SlideCheck/SlideCheck_Infer.py \
  --features_h5 features.h5 --ckpt best.pt --out_json out.json

# Legacy finetuning
python Finetune_SlideCheck/SlideCheck_Finetune.py \
  --pt_path train_data.pt --exp_name my_exp --epochs 200
```

---

## Acknowledgement

Thanks to the following repositories for inspiring this repository:
  - https://github.com/mahmoodlab/TRIDENT
  - https://huggingface.co/xtxx/Digepath

---

## Contributing

Personal experience is limited, and code submissions are welcome. Please feel free to open issues or pull requests.

---

## License

This project is released under the [Apache License 2.0](LICENSE).
