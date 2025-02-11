import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pipelines.transformers import apply_clahe_unsharp

import pickle
import torch
from facenet_pytorch import MTCNN
from pipelines.utils.face_detection_utils import extract_mtcnn_boxes
import yaml
import pandas as pd

with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)


if __name__ == "__main__":

    input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # train_data_filtered.pt
    output_path = os.path.join(params["data"]["processed"], sys.argv[2])  # train_data_clahe.pt
    mean_std_path = os.path.join(params["data"]["processed"], sys.argv[3])  # mean_std.csv

    num_workers = os.cpu_count()
    # Cargar datos
    filtered_data = torch.load(input_path)
    mean, std = pd.read_csv(mean_std_path, header=None).values

    # Convertir a tensores
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clahe_data = apply_clahe_unsharp(filtered_data, mean, std, clip_limit=1, tile_grid_size=(32, 32),
                                     unsharp_amount=1.2, num_workers=num_workers)

    torch.save(clahe_data, output_path)

    print(f"CLAHE + Unsharp: {output_path}")
