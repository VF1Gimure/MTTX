import os
import sys
import yaml
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pipelines.models.vgg16_classifier import VGG16Classifier
from torch.utils.data import DataLoader, random_split
from pipelines.models.cnn_custom import CNN_custom
from pipelines.utils.data_utils import TwoChannelDataset
from pipelines.models.cnn_torch_helpers import train
import time

# Load parameters from YAML
with open("params.yaml", "r") as ymlfile:
    params = yaml.safe_load(ymlfile)

if __name__ == "__main__":
    train_input_path = os.path.join(params["data"]["processed"], sys.argv[1])  # train_data_2channel.pt
    model_output_path = os.path.join(params["data"]["models"], sys.argv[2])  # model.pth

    # Load processed data (saved as a dictionary)
    train_data = torch.load(train_input_path)
    train_dataset = TwoChannelDataset(train_data)

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    epochs = 10
    lr = 1e-4

    # Model Definition
    model = VGG16Classifier()
    for param in model.vgg.features.parameters():
        param.requires_grad = False
    for param in model.vgg.classifier.parameters():
        param.requires_grad = True


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    start_time = time.time()
    _, epochs_h = train(model, optimizer, train_loader, epochs, 'cpu')
    training_time = time.time() - start_time

    # Save trained model
    torch.save(model.state_dict(), model_output_path)

    with open(model_output_path.replace(".pth", "_time.txt"), "w") as f:
        f.write(f"Training Time: {training_time:.4f} seconds")

    print(f"Model saved at: {model_output_path}")