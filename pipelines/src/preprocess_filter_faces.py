import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pickle
import torch
from pipelines.utils.face_detection_utils import filter_profile_faces_box
import yaml
import pandas as pd

with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)


if __name__ == "__main__":
    input_mtcnn_boxes = os.path.join(params["data"]["processed"], sys.argv[1])  # mtcnn_boxes.pkl
    input_train_data = os.path.join(params["data"]["processed"], sys.argv[2])  # train_data.pt
    output_files = [os.path.join(params["data"]["processed"], f) for f in sys.argv[3:]]
    # [filtered_boxes.pkl, filtered_angles.pkl]

    # Load MTCNN boxes
    with open(input_mtcnn_boxes, "rb") as f:
        filter_boxes = pickle.load(f)

    # Load train data
    loaded_data = torch.load(input_train_data)

    # Extract image sizes
    image_sizes = {idx: tensor.shape[-2:] for idx, tensor in zip(loaded_data["idxs"], loaded_data["tensors"])}

    # Filter faces based on angle
    filtered_boxes, filtered_angles, _ = filter_profile_faces_box(filter_boxes, image_sizes,
                                                                                 angle_threshold=60)

    # Save filtered data
    for data, path in zip([filtered_boxes, filtered_angles], output_files):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    print(f"Filtrando rostros en: {output_files}")