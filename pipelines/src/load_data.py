import os
import sys

import pandas as pd
import torch
from torchvision import datasets, transforms
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pipelines.utils.transformers_setup import clahe_unsharp_t
from pipelines.utils.data_utils import load_data_image_folder


if __name__ == "__main__":
    data_path = sys.argv[1]
    output_file = sys.argv[2]
    num_workers = os.cpu_count()

    default_transform = clahe_unsharp_t() #TODO: This needs to be LOADED

    loaded_data = load_data_image_folder(data_path, default_transform,num_workers)

    torch.save(loaded_data, output_file)
