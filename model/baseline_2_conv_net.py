import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineConvNet(nn.Module):
    def __init__(self, num_classes, input_size=(128, 128), in_channels=2, channel1=16, channel2=32):
        """
        Baseline ConvNet for multiclass classification.

        Args:
            num_classes (int): Number of output classes.
            input_size (tuple): Size of the input image (H, W).
            in_channels (int): Number of input channels (default: 2).
            channel1 (int): Number of filters in the first conv layer.
            channel2 (int): Number of filters in the second conv layer.
        """
        super(BaselineConvNet, self).__init__()

        # Lambda for creating convolutional layers with kernel_size=3 and padding=1
        conv_k3 = lambda in_channels, out_channels: nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Convolutional layers
        self.conv1 = conv_k3(in_channels, channel1)
        self.conv2 = conv_k3(channel1, channel2)

        # Calculate the flattened size after convolutions
        self.flatten_size = channel2 * input_size[0] * input_size[1]  # No pooling means size remains the same

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the tensor
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))

        return x
