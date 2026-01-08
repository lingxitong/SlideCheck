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

- **normal vs. abnormal** (`logit_abn`)
- **noncancer vs. cancer** (`logit_can`)

---
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

## :memo: **Overall Introduction**
### :hammer: Pretrained Model Build Pipeline

SlideCheck is trained as a **supervised distribution prior** for SSL patch selection.  
We build a large-scale multi-task patch dataset by unifying multiple public ROI datasets, mapping heterogeneous multi-class labels into two binary factors:

- **normal vs. abnormal**
- **noncancer vs. cancer**

#### 1) Dataset Collection & Label Mapping

We integrate the public datasets listed below (see the `example_dataset/` directory):

- CRC_100K  
- ESCA_TCGA  
- RenalCell  
- Spider_Breast  
- Spider_Colorectal  
- Spider_Skin  
- Spider_Thorax  
- TCGA_Uniform  

All dataset metadata (paths, label mapping rules, dataset names) are defined in:

- `SlideCheck/Dataset_Preprocess/example_dataset/SlideCheck_Map_Dataset_Info.json`

After mapping, the overall label distribution is:

```json
"ALL_Dataset": {
  "normal_num": 250098,
  "abnormal_num": 609281,
  "cancer_num": 420399,
  "non_cancer_num": 438980
}
```

#### 2) Feature Extraction with Foundation Models

Instead of training on raw images directly, **SlideCheck is trained on patch embeddings** extracted by off-the-shelf pathology foundation models. We currently support three backbones:

- **UNI v1**
- **GigaPath**
- **Virchow2**

Feature extraction is implemented in:

- `SlideCheck/Dataset_Preprocess/SlideCheck_Dataset_Preprocess.py`
---

#### 3) Dual-Head MLP Training

Given extracted features, we train a simple and efficient classifier:

- **MLP backbone**
- **Two classification heads**
  - abnormal head (`logit_abn`)
  - cancer head (`logit_can`)

This design makes SlideCheck lightweight, fast to train, and easy to plug into any SSL pipeline as a patch scoring prior.

---

#### 4) Internal Validation Performance 
| Backbone | Task | Acc | BAcc | AUC | AUPRC | Sensitivity | Specificity | 
|---|---|---:|---:|---:|---:|---:|---:|
| UNI v1 | Abnormal | 0.9921 | 0.9911 | 0.9996 | 0.9999 | 0.9934 | 0.9888 | 
| UNI v1 | Cancer | 0.9928 | 0.9928 | 0.9997 | 0.9997 | 0.9918 | 0.9938 | 
| GigaPath | Abnormal | 0.9918 | 0.9910 | 0.9996 | 0.9998 | 0.9929 | 0.9890 | 
| GigaPath | Cancer | 0.9929 | 0.9929 | 0.9997 | 0.9997 | 0.9920 | 0.9938 | 
| Virchow2 | Abnormal | 0.9933 | 0.9926 | 0.9997 | 0.9999 | 0.9943 | 0.9909 | 
| Virchow2 | Cancer | 0.9935 | 0.9935 | 0.9998 | 0.9998 | 0.9940 | 0.9930 |

**Notes:**  
- *Abnormal* refers to **normal vs. abnormal** head.  
- *Cancer* refers to **noncancer vs. cancer** head.  


### :key: Inference Pipeline

The inference script is located at:

- `SlideCheck/Infer_SlideCheck/SlideCheck_Infer.py`

It takes a patch feature `.h5` file as input, loads a SlideCheck checkpoint, and writes predictions to a JSON file.

---

#### Usage

```bash
python SlideCheck/Infer_SlideCheck/SlideCheck_Infer.py \
  --features_h5 /path/to/features.h5 \
  --ckpt /path/to/best.pt \
  --out_json /path/to/out.json \
  --in_dim 1024 \
  --device cuda:0
```

### :jack_o_lantern: Finetune Pipeline

Due to **domain generalization** issues in histopathology (scanner/stain/site shifts), we recommend performing **lightweight continual learning / few-epoch finetuning** on your private dataset before using SlideCheck for large-scale scoring or sampling.

The finetune script is located at:

- `SlideCheck/Finetune_SlideCheck/SlideCheck_Finetune.py`

This script trains a **dual-head classifier** (`normal vs abnormal`, `noncancer vs cancer`) on pre-extracted feature embeddings stored in a `.pt` file, with optional constraint regularization (`cancer => abnormal`) and automatic experiment logging / checkpointing.

---

#### Usage

```bash
python SlideCheck/Finetune_SlideCheck/SlideCheck_Finetune.py \
  --pt_path /path/to/train_data.pt \
  --log_root_dir ./Logs_SlideCheck \
  --exp_name my_finetune \
  --device cuda:0 \
  --model_tag mlp_v1 \
  --hidden_dim 768 \
  --dropout 0.1 \
  --epochs 200 \
  --batch_size 1000000 \
  --lr 1e-3 \
  --weight_decay 1e-4
  --resume_model_ckpt pretrained_slidecheck.pt
```


## :beers: **Acknowledgement**
Thanks to the following repositories for inspiring this repository
  - https://github.com/mahmoodlab/TRIDENT
  - https://huggingface.co/xtxx/Digepath

## :sparkles: **Git Pull**
Personal experience is limited, and code submissions are welcome.