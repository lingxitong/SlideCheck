import os
import argparse
import torch
import warnings
from model_utils.model_factory import encoder_factory
from process_utils.extract_patch_features import extract_patch_features_from_slidecheck_dataloader
from dataset_utils.roi_dataset import SlideCheck_ROIDataSet
from torchvision import transforms
warnings.filterwarnings("ignore")
    

def modify_transforms(original_transform, new_resize_size):
    """
    Modify transform:
    1. Replace Resize transform size with new_resize_size
    2. Remove CenterCrop transform
    
    Args:
        original_transform: Original transform (usually a Compose object)
        new_resize_size: New resize size
    
    Returns:
        Modified transform
    """
    # If it's a Compose object, get the transforms list
    if isinstance(original_transform, transforms.Compose):
        transform_list = original_transform.transforms
    else:
        # If not Compose, wrap it as a list
        transform_list = [original_transform]
    
    new_transforms = []
    
    for t in transform_list:
        # Skip CenterCrop
        if isinstance(t, transforms.CenterCrop):
            print(f"  - Removing CenterCrop transform")
            continue
        
        # Replace Resize transform size
        if isinstance(t, transforms.Resize):
            print(f"  - Replacing Resize: {t.size} -> {new_resize_size}")
            # Keep original interpolation method and other parameters
            new_t = transforms.Resize(
                (new_resize_size,new_resize_size),
                interpolation=t.interpolation if hasattr(t, 'interpolation') else transforms.InterpolationMode.BILINEAR,
                max_size=t.max_size if hasattr(t, 'max_size') else None,
                antialias=t.antialias if hasattr(t, 'antialias') else None
            )
            new_transforms.append(new_t)
        else:
            # Keep other transforms
            new_transforms.append(t)
    
    return transforms.Compose(new_transforms)


def main(args):
    # Setup device and model
    device = torch.device(args.device)
    model = encoder_factory(args.model_name)
    
    # Get original transforms and modify
    data_transform = modify_transforms(model.eval_transforms, args.resize_size)
    model = model.to(device)
    model.eval()
    
    # Prepare data transforms and datasets
    slidecheck_roidataset = SlideCheck_ROIDataSet(args.dataset_csv,data_transform,args.dataset_class_csv)
    slidecheck_roidataloader = torch.utils.data.DataLoader(
        slidecheck_roidataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=SlideCheck_ROIDataSet.collate_fn)
    
    # Extract features
    print("Extracting features...")
    features = extract_patch_features_from_slidecheck_dataloader(model, slidecheck_roidataloader, device, args.model_name)
    # Create save directory and save features
    os.makedirs(args.save_dir, exist_ok=True)
    dataset_name = os.path.basename(args.dataset_csv).replace('.csv','')
    save_path = os.path.join(args.save_dir, f'Dataset_[{dataset_name}]_Model_[{args.model_name}]_Size_[{args.resize_size}].pt')
    torch.save(features, save_path)
    print("Feature extraction completed!")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ROI features from dataset')
    
    # Dataset parameters
    parser.add_argument('--dataset_csv', type=str, 
                        default='SlideCheck/Dataset_Preprocess/example_dataset/CRC_100K/CRC-100K.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--dataset_class_csv', type=str, 
                        default='/mnt/sdb/lxt/SlideCheck/SlideCheck/Dataset_Preprocess/example_dataset/CRC_100K/CRC-100K-Class.csv',
                        help='Path to class to ID mapping file')
    # Model parameters
    parser.add_argument('--model_name', type=str, default='uni_v1',
                        help='Model name')
    parser.add_argument('--resize_size', type=int, default=224,
                        help='Image resize size')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device ID (e.g., cuda:0 or cpu)')
    # Save path
    parser.add_argument('--save_dir', type=str, 
                        default='SlideCheck/Dataset_Features',
                        help='Directory to save extracted features')
    args = parser.parse_args()
    # Run main function
    main(args)