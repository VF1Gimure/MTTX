from torchvision import datasets, transforms
from ..transformers.torch_clahe import CLAHEUnsharpTransform


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