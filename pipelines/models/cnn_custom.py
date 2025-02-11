import torch.nn as nn
import torch.nn.functional as F

conv_k3 = lambda in_channels, out_channels: nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class CNN_custom(nn.Module):
    def __init__(self, in_channel, channel1, channel2):
        super().__init__()  # initialize the super -> nn.Module
        self.conv1 = conv_k3(in_channel, channel1)  # We are reusing our conv_k3 for ease
        self.bn1 = nn.BatchNorm2d(channel1)
        # Batch normalization after a convolution layer so we can normalize the output

        self.conv2 = conv_k3(channel1, channel2)
        self.bn2 = nn.BatchNorm2d(channel2)

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn2(self.conv2(
            F.relu(self.bn1(self.conv1(x))))))
        #print("Shape after CNN block:", x.shape)  # Debug print

        return self.max_pool(x)