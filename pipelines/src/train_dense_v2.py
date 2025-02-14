import os
import sys
import yaml
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
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

# Load parameters from YAML
with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    train_input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # Dataset file
    model_name = sys.argv[2]  # Model name identifier
    model_output_path = os.path.join(params["data"]["models"], f"{model_name}.pth")

    # Load dataset
    train_data = torch.load(train_input_path)
    train_dataset = TwoChannelDataset(train_data)
    batch_size = params["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    epochs = params["models"][model_name]["epochs"]
    lr = float(params["training"]["learning_rate"])
    img_size = (params["image"]["size_x"],params["image"]["size_y"])

    model_dict = {
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
        )
    }
    print(
        f"Model: {model_name}, Growth Rate: {params['models'][model_name]['growth_rate']}, Num Layers: {params['models'][model_name]['num_layers']}")

    model = model_dict[model_name]
    device = "mps"

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    start_time = time.time()
    _, epochs_h = train(model, optimizer, train_loader, epochs, device)
    training_time = time.time() - start_time

    torch.save(model.state_dict(), model_output_path)

    with open(model_output_path.replace(".pth", "_time.txt"), "w") as f:
        f.write(f"Training Time: {training_time:.4f} seconds")

    print(f"Model saved at: {model_output_path}")