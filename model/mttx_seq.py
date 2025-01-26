import torch
import torch.nn as nn
import torch.nn.functional as F

# Conv function for simplicity (same as before)
conv_k3 = lambda in_channels, out_channels: nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
class MTTX_Seq(nn.Module):
    def __init__(self, in_channel, channel1=16, channel2=32, channel3=64, channel4=128):
        super(MTTX_Seq, self).__init__()

        # First convolution block
        self.conv1 = conv_k3(in_channel, channel1)  # 1 -> 16 channels (or input channels as needed)
        self.bn1 = nn.BatchNorm2d(channel1)  # Batch normalization after conv1
        self.conv2 = conv_k3(channel1, channel2)  # 16 -> 32 channels
        self.bn2 = nn.BatchNorm2d(channel2)  # Batch normalization after conv2

        # Second convolution block
        self.conv3 = conv_k3(channel2, channel3)  # 32 -> 64 channels
        self.bn3 = nn.BatchNorm2d(channel3)  # Batch normalization after conv3
        self.conv4 = conv_k3(channel3, channel4)  # 64 -> 128 channels
        self.bn4 = nn.BatchNorm2d(channel4)  # Batch normalization after conv4

        # MaxPooling layer (this will reduce dimensions)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # Apply convolutions, batch normalization, ReLU activation, and max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # conv1 + batchnorm + pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # conv2 + batchnorm + pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # conv3 + batchnorm + pool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # conv4 + batchnorm + pool
        return x  # Return output after pooling