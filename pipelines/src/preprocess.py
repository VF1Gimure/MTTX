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

import os
import sys
import torch
import yaml
from torch.utils.data import random_split
from pipelines.utils.data_utils import TwoChannelDataset, tensors_to_2_channels, crop_faces, resize_faces

# Load parameters from params.yaml
with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)


def transform_2_to_3_channels(data):
    transformed_tensors = []

    for t in data["tensors"]:
        if t.dim() == 3:  # Shape: (C, H, W), should be (2, H, W)
            t = t.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)

        # Ensure exactly 2 channels exist before copying
        if t.shape[1] == 2:
            t = torch.cat([t, t[:, 1:2, :, :]], dim=1)  # Copy channel 2 as channel 3

        transformed_tensors.append(t.squeeze(0))  # Remove batch dimension

    return {
        "tensors": transformed_tensors,
        "labels": data["labels"]
    }


if __name__ == "__main__":

    input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # train_data_landmarked.pt
    output_train_2c = os.path.join(params["data"]["processed"], sys.argv[2])  # train_data_2channel.pt
    output_test_2c = os.path.join(params["data"]["processed"], sys.argv[3])  # test_data_2channel.pt
    output_train_3c = os.path.join(params["data"]["processed"], sys.argv[4])  # train_data_3channel.pt
    output_test_3c = os.path.join(params["data"]["processed"], sys.argv[5])  # test_data_3channel.pt

    num_workers = os.cpu_count()

    # Load landmarked data
    landmarked_data = torch.load(input_path)

    # Apply processing
    cropped = crop_faces(landmarked_data, padding=5, num_workers=num_workers)
    resized = resize_faces(cropped, (224, 224), num_workers)
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

    # Save both 2-channel and 3-channel versions
    torch.save(train_data, output_train_2c)
    torch.save(test_data, output_test_2c)

    torch.save(transform_2_to_3_channels(train_data), output_train_3c)
    torch.save(transform_2_to_3_channels(test_data), output_test_3c)

    print(f"Train dataset saved at: {output_train_2c}")
    print(f"Test dataset saved at: {output_test_2c}")

