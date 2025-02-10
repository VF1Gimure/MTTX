from torchvision import datasets, transforms
from ..transformers.torch_clahe import CLAHEUnsharpTransform

import torch
from tqdm import tqdm

def clahe_unsharp_t(clip_limit=1.0, tile_grid_size=(32, 32), unsharp_amount=1.2, mean=None, std=None):
    if std is None:
        std = [0.5]
    if mean is None:
        mean = [0.5]
    data_transforms = transforms.Compose([
        CLAHEUnsharpTransform(clip_limit=clip_limit, tile_grid_size=tile_grid_size, unsharp_amount=unsharp_amount),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return data_transforms


import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm

def compute_mean_std(dataset_path):
    """
    Computes per-channel mean and std for the dataset with variable image sizes.

    Args:
        dataset_path (str): Path to dataset folder.

    Returns:
        tuple: (mean, std) for each channel.
    """
    dataset = datasets.ImageFolder(dataset_path, transform=transforms.ToTensor())

    mean = torch.zeros(3)
    std = torch.zeros(3)
    count = 0

    for idx, (img, _) in tqdm(enumerate(dataset), total=len(dataset), desc="Computing Mean & Std"):
        # Compute mean and std for each image separately
        mean += img.mean(dim=(1, 2))
        std += img.std(dim=(1, 2))
        count += 1

    mean /= count
    std /= count

    return mean.tolist(), std.tolist()


def transform_tensor(mean=None, std=None):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
