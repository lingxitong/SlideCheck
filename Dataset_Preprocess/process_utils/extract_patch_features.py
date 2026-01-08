import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

@torch.no_grad()
def extract_patch_features_from_dataloader(model, dataloader,device,model_name):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, (2) [N x 1]-dim np.array of labels, and (3) list of image names

    """
    all_embeddings, all_labels, all_img_names = [], [], []
    batch_size = dataloader.batch_size

    for batch_idx, (batch, target, img_names) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        
        # For summitconch model, need to convert input data to bfloat16
        with torch.inference_mode():
            if model_name == 'virchow_v2':
                embeddings = model(batch)
                class_token = embeddings[:, 0]    
                patch_tokens = embeddings[:, 5:] 
                embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                embeddings = embeddings.detach().cpu()[:remaining, :].cpu()
            else:
                embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_img_names.extend(img_names[:remaining])

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
        "img_names": all_img_names,
    }

    return asset_dict


@torch.no_grad()
def extract_patch_features_from_slidecheck_dataloader(model, dataloader, device, model_name):
    """
    Returns:
        dict with keys:
            embeddings: [N, D] float32
            glob_labels: [N]
            normal_abnormal_labels: [N]
            cancer_noncancer_labels: [N]
            img_names: list[str] length N
    """
    all_embeddings = []
    all_glob_labels = []
    all_normal_labels = []
    all_cancer_labels = []
    all_img_names = []

    batch_size = dataloader.batch_size

    for batch_idx, (batch, glob_labels, normal_labels, cancer_labels, img_names) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]

        # Pad last batch (keep your original logic)
        if remaining != batch_size:
            pad = torch.zeros((batch_size - remaining,) + batch.shape[1:], dtype=batch.dtype, device=batch.device)
            batch = torch.vstack([batch, pad])

        batch = batch.to(device)

        with torch.inference_mode():
            if model_name == "virchow_v2":
                embeddings = model(batch)
                class_token = embeddings[:, 0]
                patch_tokens = embeddings[:, 5:]
                embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                embeddings = embeddings[:remaining].detach().cpu()
            else:
                embeddings = model(batch)[:remaining].detach().cpu()

            # labels -> numpy (只取真实 remaining 部分)
            glob_np = glob_labels[:remaining].detach().cpu().numpy()
            normal_np = normal_labels[:remaining].detach().cpu().numpy()
            cancer_np = cancer_labels[:remaining].detach().cpu().numpy()

            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_glob_labels.append(glob_np)
        all_normal_labels.append(normal_np)
        all_cancer_labels.append(cancer_np)
        all_img_names.extend(list(img_names[:remaining]))

    asset_dict = {
        "embeddings": np.vstack([e.numpy() for e in all_embeddings]).astype(np.float32),
        "glob_labels": np.concatenate(all_glob_labels),
        "normal_abnormal_labels": np.concatenate(all_normal_labels),
        "cancer_noncancer_labels": np.concatenate(all_cancer_labels),
        "img_names": all_img_names,
    }
    return asset_dict