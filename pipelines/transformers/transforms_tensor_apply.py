from torchvision import datasets
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pipelines.utils.data_utils import denormalize
from pipelines.transformers.torch_clahe import CLAHEUnsharpTransform
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms

def apply_clahe_unsharp(loaded_data, mean, std, clip_limit=2.0, tile_grid_size=(8, 8), unsharp_amount=1.5, num_workers=8):
    """
    Applies CLAHE + Unsharp Masking (Grayscale) to all tensors in `loaded_data` using multithreading.

    Args:
        loaded_data (dict): Dataset containing tensors and metadata.
        mean (torch.Tensor): Normalization mean values.
        std (torch.Tensor): Normalization std values.
        clip_limit (float): CLAHE contrast limit.
        tile_grid_size (tuple): CLAHE tile grid size.
        unsharp_amount (float): Strength of unsharp masking.
        num_workers (int): Number of parallel workers.

    Returns:
        dict: Updated dataset with enhanced grayscale tensors.
    """
    clahe_unsharp = CLAHEUnsharpTransform(
        clip_limit=clip_limit, tile_grid_size=tile_grid_size, unsharp_amount=unsharp_amount
    )

    def transform_tensor(tensor):
        """Denormalize, apply CLAHE + UNSHARP, convert back to tensor, and re-normalize."""
        # Step 1: Denormalize tensor
        tensor_denorm = denormalize(tensor, mean, std)
        # Step 2: Convert to PIL image
        image_pil = to_pil_image(tensor_denorm)

        # Step 3: Apply CLAHE + UNSHARP (Grayscale)
        transformed_image = clahe_unsharp(image_pil)

        # Step 4: Convert back to Tensor (single-channel grayscale)
        transformed_tensor = transforms.ToTensor()(transformed_image)  # Shape: (1, H, W)

        # Grayscale
        mean_gray = mean.mean()
        std_gray = std.mean()

        transformed_tensor = (transformed_tensor - mean_gray) / std_gray  # Normalize

        return transformed_tensor

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        enhanced_tensors = list(tqdm(
            executor.map(transform_tensor, loaded_data["tensors"]),
            total=len(loaded_data["tensors"]),
            desc="Aplicando CLAHE + UNSHARP (Grayscale) a Tensores"
        ))

    # Update dataset with enhanced grayscale tensors
    loaded_data["tensors"] = enhanced_tensors
    return loaded_data