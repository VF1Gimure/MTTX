import os
import sys

import pandas as pd
from torch.utils.data import random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
from pipelines.utils.data_utils import load_data_image_folder, TwoChannelDataset, tensors_to_2_channels, resize_faces, \
    crop_faces
from pipelines.utils import (
    transform_tensor,compute_mean_std)

import yaml

with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)



if __name__ == "__main__":

    input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # train_data_landmarked.pt
    output_train_path = os.path.join(params["data"]["processed"], sys.argv[2])  # train_data_2channel.pt
    output_test_path = os.path.join(params["data"]["processed"], sys.argv[3])  # test_data_2channel.pt

    num_workers = os.cpu_count()
    # Cargar datos
    landmarked_data = torch.load(input_path)

    cropped = crop_faces(landmarked_data, padding=5, num_workers=num_workers)
    resized = resize_faces(cropped, (512, 512), num_workers)
    training_data = tensors_to_2_channels(resized, 1, num_workers)

    # Create Dataset
    dataset = TwoChannelDataset(training_data)

    # Split dataset
    split_ratio = 0.8
    dataset_length = len(dataset)
    train_length = int(split_ratio * dataset_length)
    test_length = dataset_length - train_length

    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])

    train_data = {
        "tensors": [dataset.tensors[i] for i in train_dataset.indices],
        "labels": [dataset.labels[i] for i in train_dataset.indices]
    }
    test_data = {
        "tensors": [dataset.tensors[i] for i in test_dataset.indices],
        "labels": [dataset.labels[i] for i in test_dataset.indices]
    }

    torch.save(train_data, output_train_path)
    torch.save(test_data, output_test_path)

    print(f"Train dataset saved at: {output_train_path}")
    print(f"Test dataset saved at: {output_test_path}")

