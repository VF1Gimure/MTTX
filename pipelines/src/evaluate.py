import os
import sys
import yaml
import torch
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from torch.utils.data import DataLoader
from pipelines.models.cnn_custom import CNN_custom
from pipelines.utils.data_utils import TwoChannelDataset
from pipelines.models.cnn_torch_helpers import evaluate, accuracy
from pipelines.utils.metrics import compute_classification_metrics, normal_cm, get_predictions_and_probs

# Load parameters from YAML
with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    test_input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # test_data_2channel.pt
    model_input_path = os.path.join(params["data"]["models"], sys.argv[2])  # model.pth
    results_output_path = os.path.join(params["data"]["processed"], sys.argv[3])  # metrics.csv
    conf_matrix_output_path = os.path.join(params["data"]["processed"], sys.argv[4])  # conf_matrix.pkl

    # Load test dataset
    test_data = torch.load(test_input_path)
    test_dataset = TwoChannelDataset(test_data)
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model Configuration
    channels = 2
    channel_1 = 32
    channel_2 = 64
    resize_input = (512, 512)
    flattened_size = channel_2 * int((resize_input[1] / 2) / 2) * int((resize_input[0] / 2) / 2)
    out_features = 8

    # Load trained model
    model_base = torch.nn.Sequential(
        CNN_custom(channels, channel_1, channel_1),
        CNN_custom(channel_1, channel_2, channel_2),
        torch.nn.Flatten(),
        torch.nn.Linear(flattened_size, out_features),
    )
    model_base.load_state_dict(torch.load(model_input_path))
    model_base.eval()

    # Run evaluation
    predictions, true_labels = evaluate(model_base, test_loader, device="mps")

    # Compute confusion matrix
    label_map = {0: "angry", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
    label_names = list(label_map.values())
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Save confusion matrix (for later visualization)
    with open(conf_matrix_output_path, "wb") as f:
        pickle.dump(conf_matrix, f)

    # Compute Classification Metrics
    df_clases, df_macro = compute_classification_metrics(conf_matrix, label_map)
    df_macro.loc["Accuracy"] = accuracy(predictions, true_labels)

    # Save metrics
    df_macro.to_csv(results_output_path)

    print("\n====== Per-Class Metrics ======")
    print(df_clases)
    print("\n====== Macro Metrics ======")
    print(df_macro.to_string())

    print(f"Evaluation results saved at: {results_output_path}")
    print(f"Confusion matrix saved at: {conf_matrix_output_path}")
