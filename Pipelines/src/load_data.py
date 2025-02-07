import os
import sys

import pandas as pd
import torch
from torchvision import datasets, transforms
from Pipelines.utils.transformers_setup import clahe_unsharp_t
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def load_data_image_folder(dataset_path, transformer, num_workers):
    i_dataset = datasets.ImageFolder(dataset_path, transform=transformer)

    # Extract data in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda idx: (i_dataset[idx][0], i_dataset[idx][1], i_dataset.imgs[idx][0]),
                         range(len(i_dataset))),
            total=len(i_dataset),
            desc="Extracting Tensors, Targets, and File Paths..."
        ))

    # Unpack results into separate lists
    tensors, labels, file_paths = zip(*results)

    return {
        "tensors": torch.stack(tensors),  # Stack tensors into a single tensor batch
        "labels": torch.tensor(labels),  # Convert labels to tensor
        "file_paths": list(file_paths)  # Keep file paths as a list
    }


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_file = sys.argv[2]
    num_workers = os.cpu_count()

    default_transform = clahe_unsharp_t()
    print(f"[INFO] Data TO LOAD{data_path}")

    #loaded_data = load_data_image_folder(data_path, default_transform,num_workers)
    print(f"[INFO] Processed tensors and metadata saved to {output_file}")

    #torch.save(loaded_data, output_file)
