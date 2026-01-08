# SlideCheck 
#### SlideCheck as Prior: Guiding Self-Supervised Pathology Representation Learning with Dataset Distributions 
<p align="center">
  <a href='https://arxiv.org/abs/2505.21928'>
  <img src='https://img.shields.io/badge/Arxiv-2404.19759-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://huggingface.co/xtxx/SlideCheck'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow'></a>
  <a href='https://lingxitong/SlideCheck'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
</p>

<img src="https://github.com/lingxitong/SlideCheck/blob/main/SlideCheck_Logo.png"  width="390px" align="right" />
Self-supervised learning (SSL) has shown strong transferability for pathology foundation models, yet most pipelines still sample patches from whole-slide images (WSIs) uniformly at random despite severe redundancy and imbalanced tissue distributions. We propose SlideCheck as a prior, using supervised distribution priors to guide SSL patch selection. We unify multiple large-scale public ROI datasets and map heterogeneous labels into two binary factors: normal vs. abnormal and cancer vs. non-cancer. With ~1M labeled patches, we train and open-source SlideCheck, a lightweight patch classifier that outputs prior scores for candidate patches. These scores can be used to filter and prioritize diagnostically relevant patches before or during SSL pretraining, reducing uninformative tissue redundancy and improving data efficiency without changing the SSL objective. We hope SlideCheck can serve as a practical, reusable tool to facilitate dataset curation and patch sampling for future pathology SSL research. SlideCheck is a lightweight Foundation Model Based dual-head patch classifier that outputs two signals for each patch feature:

- **Normal vs. ABNormal** (`logit_abn`)
- **NonCancer vs. Cancer** (`logit_can`)

## Repository Structure (Example)

```text
.
├── SlideCheck_Model.py
├── infer_slidecheck.py
└── README.md
```

---

## Input Format

The inference script expects an HDF5 file containing a dataset named `features` by default.

* **HDF5 key**: `features` (changeable via `--h5_key`)
* **Shape**: `[N, D]` (recommended) or `[B, D]`
* **D must match** `--in_dim`

Example structure inside HDF5:

```text
/features   float32 [N, D]
```

---

## Inference

Basic usage:

```bash
python infer_slidecheck.py \
  --features_h5 /path/to/features.h5 \
  --ckpt /path/to/best.pt \
  --out_json /path/to/out.json \
  --model_tag mlp_v1 \
  --in_dim 1024 \
  --device cuda:0 \
  --threshold 0.5
```

If your HDF5 dataset key is not `features`:

```bash
python infer_slidecheck.py \
  --features_h5 /path/to/features.h5 \
  --h5_key feats \
  --ckpt /path/to/best.pt \
  --out_json /path/to/out.json \
  --model_tag mlp_v1 \
  --in_dim 1024 \
  --device cuda:0
```

---

## Output JSON

The script produces a JSON file like:

```json
{
  "threshold": 0.5,
  "logit_abn_binary": [0, 1, 0, 0],
  "logit_can_binary": [0, 0, 1, 0]
}
```

* `logit_abn_binary[i] = 1` means the *i-th patch* is predicted **abnormal**
* `logit_can_binary[i] = 1` means the *i-th patch* is predicted **cancer**

### Save sigmoid probabilities (optional)

If you enable `--save_probs`, the JSON will also include:

* `logit_abn`: sigmoid probabilities in `[0, 1]`
* `logit_can`: sigmoid probabilities in `[0, 1]`

Example:

```json
{
  "threshold": 0.5,
  "logit_abn_binary": [0, 1, 0, 0],
  "logit_can_binary": [0, 0, 1, 0],
  "logit_abn": [0.12, 0.83, 0.33, 0.09],
  "logit_can": [0.05, 0.21, 0.77, 0.11]
}
```

---

## Notes

* This inference script runs **single-pass** on the entire feature tensor (no batch_size).
* `--in_dim` must match the feature dimension `D` stored in the HDF5 file.
* Checkpoints are expected to be either:

  * a dict containing `state_dict`, or
  * a raw `state_dict`

---

## License

Add your license here (e.g., MIT / Apache-2.0).

```
```
