from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

def load_class2id_mapping(filepath):
    class2id = {}
    with open(filepath, 'r') as file:
        for line in file:
            class_name, class_id = line.strip().split(',')
            class2id[class_name] = int(class_id)
    return class2id

class ROIDataSet(Dataset):
    """Custom dataset for ROI"""

    def __init__(self, csv_path,domain,transform,class2id_txt):
        self.csv_path = csv_path
        self.domain = domain
        self.df = pd.read_csv(csv_path)
        self.images_path = self.df[f'{domain}_path'].dropna().tolist()
        self.images_class = self.df[f'{domain}_label'].dropna().tolist()
        self.transform = transform
        self.class2id_dict = load_class2id_mapping(class2id_txt)
        self.id2class_dict = {v: k for k, v in self.class2id_dict.items()}

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path)
        # RGB for color images, L for grayscale images
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(img_path))
        label = self.images_class[item]
        img_name = os.path.basename(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label), img_name
    
    def get_imgs_from_idxs(self, idxs):
        img_paths = [self.images_path[idx] for idx in idxs]
        labels = [self.images_class[idx] for idx in idxs]
        return img_paths, labels

    @staticmethod
    def collate_fn(batch):
        # Reference official default_collate implementation
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, img_names = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, img_names
    
class SlideCheck_ROIDataSet(Dataset):
    """Custom dataset for ROI"""

    def __init__(self, csv_path, transform, class2id_csv):
        self.csv_path = csv_path
        self.class2id_csv = class2id_csv

        self.df = pd.read_csv(csv_path)
        self.images_path = self.df["img_path"].dropna().tolist()
        self.origin_images_labels = self.df["img_label"].dropna().tolist()  # 原始 cls_id
        self.transform = transform

        # 生成 clsid -> normal/abnormal 和 clsid -> cancer/noncancer 映射
        self.get_label_dict()

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path)

        if img.mode != "RGB":
            raise ValueError(f"image: {img_path} isn't RGB mode.")

        global_label = int(self.origin_images_labels[item])  # 原始多类 label
        normal_abnormal_label = int(self.clsid2normal_abnormal[global_label])  # 0/1
        cancer_noncancer_label = int(self.clsid2cancer_noncancer[global_label])  # 0/1

        img_name = os.path.basename(img_path)

        if self.transform is not None:
            img = self.transform(img)

        # 返回三个 label
        return img, global_label, normal_abnormal_label, cancer_noncancer_label, img_name

    def get_imgs_from_idxs(self, idxs):
        img_paths = [self.images_path[idx] for idx in idxs]
        global_labels = [int(self.origin_images_labels[idx]) for idx in idxs]
        normal_labels = [int(self.clsid2normal_abnormal[g]) for g in global_labels]
        cancer_labels = [int(self.clsid2cancer_noncancer[g]) for g in global_labels]
        return img_paths, global_labels, normal_labels, cancer_labels

    def get_label_dict(self):
        df = pd.read_csv(self.class2id_csv)

        # 规范化字符串（防止大小写/空格问题）
        df["normal_abnormal"] = df["normal_abnormal"].astype(str).str.strip().str.lower()
        df["cancer_noncancer"] = df["cancer_noncancer"].astype(str).str.strip().str.lower()
        df["cls_id"] = df["cls_id"].astype(int)

        normal_map = {"normal": 0, "abnormal": 1}
        cancer_map = {"noncancer": 0, "cancer": 1}

        self.clsid2normal_abnormal = {
            int(cid): normal_map[na]
            for cid, na in zip(df["cls_id"].tolist(), df["normal_abnormal"].tolist())
        }
        self.clsid2cancer_noncancer = {
            int(cid): cancer_map[cn]
            for cid, cn in zip(df["cls_id"].tolist(), df["cancer_noncancer"].tolist())
        }
    @staticmethod
    def collate_fn(batch):
        # batch: list of (img, global_label, normal_label, cancer_label, img_name)
        images, y_global, y_normal, y_cancer, img_names = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        y_global = torch.as_tensor(y_global, dtype=torch.long)
        y_normal = torch.as_tensor(y_normal, dtype=torch.long)
        y_cancer = torch.as_tensor(y_cancer, dtype=torch.long)
        return images, y_global, y_normal, y_cancer, img_names
    
class FeatDataSet(Dataset):
    """Custom feature dataset"""

    def __init__(self, all_feats,all_labels):
        self.all_feats = all_feats
        self.all_labels = all_labels

    def __len__(self):
        return len(self.all_feats)

    def __getitem__(self, item):
        feat = self.all_feats[item]
        label = self.all_labels[item]
        return feat, int(label)
    
    # def get_imgs_from_idxs(self, idxs):
    #     img_paths = [self.images_path[idx] for idx in idxs]
    #     labels = [self.images_class[idx] for idx in idxs]
    #     return img_paths, labels

    @staticmethod
    def collate_fn(batch):
        # Reference official default_collate implementation
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
