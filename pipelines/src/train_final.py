import os
import sys
import yaml
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from torch.utils.data import DataLoader
from pipelines.models.modular_cnn import CustomCNN
from pipelines.models.mlp import MLPMixer,MLPMixerV2
import time
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

    img_size = (params["image"]["size_x"],params["image"]["size_y"])
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

    if model_name not in model_dict:
        raise ValueError(f"Invalid model name: {model_name}. Choose from {list(model_dict.keys())}")

    model = model_dict[model_name]
    device = "mps" if model_name in mps_models else "cpu"
    # Freeze layers for transfer learning models

    if model_name in ["efficientnet", "mobilenet", "resnet18", "vgg16", "vit"]:
        for layer in params["models"][model_name]["freeze"]:
            for param in eval(f"model.{layer}.parameters()"):
                param.requires_grad = False

            # Unfreeze layers
        for layer in params["models"][model_name]["unfreeze"]:
            for param in eval(f"model.{layer}.parameters()"):
                param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    start_time = time.time()
    _, epochs_h = train(model, optimizer, train_loader, epochs, device)
    training_time = time.time() - start_time

    # Save trained model
    torch.save(model.state_dict(), model_output_path)

    with open(model_output_path.replace(".pth", "_time.txt"), "w") as f:
        f.write(f"Training Time: {training_time:.4f} seconds")

    print(f"Model saved at: {model_output_path}")