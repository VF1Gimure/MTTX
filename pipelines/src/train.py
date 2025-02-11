import os
import sys
import yaml
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from torch.utils.data import DataLoader, random_split
from pipelines.models.cnn_custom import CNN_custom
from pipelines.utils.data_utils import TwoChannelDataset
from pipelines.models.cnn_torch_helpers import train

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

    # Model Configuration
    channels = 2
    channel_1 = 32
    channel_2 = 64
    resize_input = (512, 512)
    flattened_size = channel_2 * int((resize_input[1] / 2) / 2) * int((resize_input[0] / 2) / 2)
    out_features = 8
    epochs = 10
    lr = 1e-4

    # Model Definition
    model_base = torch.nn.Sequential(
        CNN_custom(channels, channel_1, channel_1),
        CNN_custom(channel_1, channel_2, channel_2),
        torch.nn.Flatten(),
        torch.nn.Linear(flattened_size, out_features),
    )

    optimizer = torch.optim.Adam(model_base.parameters(), lr=lr)

    # Train model
    _, epochs_h = train(model_base, optimizer, train_loader, epochs, 'mps')

    # Save trained model
    torch.save(model_base.state_dict(), model_output_path)

    print(f"Model saved at: {model_output_path}")
