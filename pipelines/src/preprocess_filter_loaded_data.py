import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
import pickle
import yaml
from pipelines.utils.data_utils import filter_loaded_data, filter_redundant_angles

# Load params.yaml
with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    # Paths from arguments
    data_path = os.path.join(params["data"]["processed"], sys.argv[1])  # train_data.pt
    filtered_boxes_path = os.path.join(params["data"]["processed"], sys.argv[2])  # filtered_boxes.pkl
    filtered_angles_path = os.path.join(params["data"]["processed"], sys.argv[3])  # filtered_angles.pkl
    output_data_path = os.path.join(params["data"]["processed"], sys.argv[4])  # train_data_filtered.pt

    # Load data
    with open(filtered_boxes_path, "rb") as f:
        filtered_boxes = pickle.load(f)
    with open(filtered_angles_path, "rb") as f:
        filtered_angles = pickle.load(f)

    loaded_data = torch.load(data_path)

    final_boxes, _, _ = filter_redundant_angles(filtered_boxes, filtered_angles, loaded_data, 0.2)

    filtered_data = filter_loaded_data(loaded_data, final_boxes)

    torch.save(filtered_data, output_data_path)

    print(f"Dataset Filtrado en: {output_data_path}")
