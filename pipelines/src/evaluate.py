import os
import sys
import yaml
import torch
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from pipelines.models.modular_cnn import CustomCNN
from pipelines.models.capsule import CustomCapsuleNet, DeepCapsuleNet
from pipelines.models.dense import CustomDenseNet,DenseNet
from pipelines.models.mlp import MLPMixer
import time
from pipelines.models.effinet_b0 import EfficientNetClassifier
from pipelines.models.mobilenet_classifier import MobileNetClassifier
from pipelines.models.resnet18_classifier import ResNet18Classifier
from pipelines.models.vgg16_classifier import VGG16Classifier
from pipelines.models.vit_classifier import ViTClassifier
from pipelines.utils.data_utils import TwoChannelDataset
from pipelines.models.cnn_torch_helpers import train
from pipelines.utils.data_utils import TwoChannelDataset
from pipelines.models.cnn_torch_helpers import evaluate, accuracy
from pipelines.utils.metrics import compute_classification_metrics, normal_cm, get_predictions_and_probs
import time

# Load parameters from YAML
with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    test_input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # test_data_2channel.pt
    model_input_path = os.path.join(params["data"]["models"], sys.argv[2])  # .pth
    results_output_path = os.path.join(params["data"]["processed"], sys.argv[3])  # metrics.pkl
    conf_matrix_output_path = os.path.join(params["data"]["processed"], sys.argv[4])  # conf_matrix.pkl
    model_name = sys.argv[2].replace(".pth", "")

    # Load test dataset
    test_data = torch.load(test_input_path)
    test_dataset = TwoChannelDataset(test_data)
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine model type from filename
    img_size = (params["image"]["size_x"], params["image"]["size_y"])

    # Model selection
    model_dict = {
        "custom_cnn": CustomCNN(
            in_channels=params["models"]["custom_cnn"]["in_channels"],
            channel1=params["models"]["custom_cnn"]["channel1"],
            channel2=params["models"]["custom_cnn"]["channel2"],
            out_features=params["models"]["custom_cnn"]["out_features"],
            img_size=img_size
        ),
        "mlp": MLPMixer(
            in_channels=params["models"]["mlp"]["in_channels"],
            img_size=img_size[0],
            num_classes=params["models"]["mlp"]["num_classes"],
            patch_size=params["models"]["mlp"]["patch_size"],
            hidden_dim=params["models"]["mlp"]["hidden_dim"],
            num_layers=params["models"]["mlp"]["num_layers"]
        ),
        "deep_capsule": DeepCapsuleNet(
            in_channels=params["models"]["deep_capsule"]["in_channels"],
            conv1_out=params["models"]["deep_capsule"]["conv1_out"],
            conv2_out=params["models"]["deep_capsule"]["conv2_out"],
            capsule_dim=params["models"]["deep_capsule"]["capsule_dim"],
            num_capsules=params["models"]["deep_capsule"]["num_capsules"],
            num_classes=params["models"]["deep_capsule"]["num_classes"],
            img_size=img_size
        ),
        "capsule": CustomCapsuleNet(
            in_channels=params["models"]["capsule"]["in_channels"],
            conv_out=params["models"]["capsule"]["conv_out"],
            capsule_dim=params["models"]["capsule"]["capsule_dim"],
            num_capsules=params["models"]["capsule"]["num_capsules"],
            num_classes=params["models"]["capsule"]["num_classes"],
            img_size=img_size
        ),
        "dense": CustomDenseNet(
            in_channels=params["models"]["dense"]["in_channels"],
            growth_rate=params["models"]["dense"]["growth_rate"],
            num_layers=params["models"]["dense"]["num_layers"],
            out_features=params["models"]["dense"]["out_features"],
            img_size=img_size
        ),
        "densenet": DenseNet(
            in_channels=params["models"]["densenet"]["in_channels"],
            growth_rate=params["models"]["densenet"]["growth_rate"],
            num_layers=params["models"]["densenet"]["num_layers"],
            out_features=params["models"]["densenet"]["out_features"],
            reduction=params["models"]["densenet"]["reduction"],
            drop_rate=params["models"]["densenet"]["drop_rate"],
        ),
        "efficientnet": EfficientNetClassifier(),
        "mobilenet": MobileNetClassifier(),
        "resnet18": ResNet18Classifier(),
        "vgg16": VGG16Classifier(),
        "vit": ViTClassifier(),
    }

    model = model_dict[model_name]
    device = "mps" if model_name in ["custom_cnn", "mlp", "deep_capsule", "capsule","dense","DenseNet"] else "cpu"
    # Load model state
    model.load_state_dict(torch.load(model_input_path))

    start_time = time.time()
    predictions, true_labels = evaluate(model, test_loader, device=device)
    inference_time = time.time() - start_time

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

    # Load training time
    with open(model_input_path.replace(".pth", "_time.txt"), "r") as f:
        training_time = float(f.read().split(":")[1].strip().split()[0])

    # Save times to metrics CSV
    df_macro.loc["Training Time"] = training_time
    df_macro.loc["Inference Time"] = inference_time
    # Save metrics
    with open(results_output_path, "wb") as f:
        pickle.dump((df_clases, df_macro), f)
    print("\n====== Per-Class Metrics ======")
    print(df_clases)
    print("\n====== Macro Metrics ======")
    print(df_macro.to_string())

    print(f"Evaluation results saved at: {results_output_path}")
    print(f"Confusion matrix saved at: {conf_matrix_output_path}")
