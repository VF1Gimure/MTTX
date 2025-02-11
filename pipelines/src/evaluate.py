import os
import sys
import yaml
import torch
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from torch.utils.data import DataLoader
from pipelines.models.resnet18_classifier import ResNet18Classifier
from pipelines.models.vgg16_classifier import VGG16Classifier
from pipelines.models.vit_classifier import ViTClassifier
from pipelines.models.effinet_b0 import EfficientNetClassifier
from pipelines.models.mobilenet_classifier import MobileNetClassifier
from pipelines.models.modular_cnn import CustomCNN
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
    results_output_path = os.path.join(params["data"]["processed"], sys.argv[3])  # metrics.csv
    conf_matrix_output_path = os.path.join(params["data"]["processed"], sys.argv[4])  # conf_matrix.pkl

    # Load test dataset
    test_data = torch.load(test_input_path)
    test_dataset = TwoChannelDataset(test_data)
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine model type from filename
    model_name = os.path.basename(model_input_path).replace(".pth", "")

    # Load appropriate model
    if "resnet" in model_name:
        model = ResNet18Classifier()
    elif "vgg16" in model_name:
        model = VGG16Classifier()
    elif "vit" in model_name:
        model = ViTClassifier()
    elif "efficientnet" in model_name:
        model = EfficientNetClassifier()
    elif "mobilenet" in model_name:
        model = MobileNetClassifier()
    else:
        model = CustomCNN(in_channels=2, channel1=32, channel2=64, out_features=8, img_size=(224, 224))

    # Load model state
    model.load_state_dict(torch.load(model_input_path))
    model.eval()

    start_time = time.time()
    predictions, true_labels = evaluate(model, test_loader, device="mps")
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
    df_macro.to_csv(results_output_path)

    print("\n====== Per-Class Metrics ======")
    print(df_clases)
    print("\n====== Macro Metrics ======")
    print(df_macro.to_string())

    print(f"Evaluation results saved at: {results_output_path}")
    print(f"Confusion matrix saved at: {conf_matrix_output_path}")
