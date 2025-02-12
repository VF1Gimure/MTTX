import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomDenseNet(nn.Module):
    def __init__(self, in_channels=2, growth_rate=32, num_layers=3, out_features=8, img_size=(224, 224)):
        super(CustomDenseNet, self).__init__()

        self.layers = nn.ModuleList()
        current_channels = in_channels
        self.pool = nn.MaxPool2d(2, 2)  # Downsampling

        # Dense Layers with Growth
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(growth_rate)
            )
            self.layers.append(layer)
            current_channels += growth_rate  # DenseNet concatenates channels

        # Transition Layer
        self.final_conv = nn.Conv2d(current_channels, 64, kernel_size=1)

        # Compute Flattened Size Dynamically
        reduced_size = img_size[0] // (2 ** num_layers)  # Assuming 2x2 pooling
        flattened_size = 64 * reduced_size * reduced_size

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flattened_size, out_features)

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            out = layer(x)
            skip_connections.append(out)
            x = torch.cat(skip_connections, dim=1)  # Concatenating along channels
            x = self.pool(x)  # Reduce spatial dimensions

        x = self.final_conv(x)
        x = self.flatten(x)
        return self.fc(x)
