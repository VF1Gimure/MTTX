import os
import sys

import dlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pickle
import torch
from facenet_pytorch import MTCNN
from pipelines.utils.face_detection_utils import extract_dlib_landmarks
import yaml
import pandas as pd

with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)


if __name__ == "__main__":
    input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # train_data_clahe.pt
    output_path = os.path.join(params["data"]["processed"], sys.argv[2])  # train_data_landmarked.pt
    mean_std_path = os.path.join(params["data"]["processed"], sys.argv[3])  # mean_std.csv

    num_workers = os.cpu_count()
    # Cargar datos
    clahe_data = torch.load(input_path)
    mean, std = pd.read_csv(mean_std_path, header=None).values

    # Convertir a tensores
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    # predictor
    dlib_68_predictor = dlib.shape_predictor(params["data"]["dlib68"])

    updated_data = extract_dlib_landmarks(clahe_data, mean, std, dlib_68_predictor, num_workers=num_workers)

    torch.save(updated_data, output_path)

    print(f"Landmarks guardados en: {output_path}")

