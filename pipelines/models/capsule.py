import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def conv_output_size(input_size, kernel_sizes, strides, paddings):
    output_size = input_size

    output_size = ((output_size - kernel_sizes + (2 * paddings)) // strides) + 1  # Integer division
    return output_size


class CustomCapsuleNet(nn.Module):
    def __init__(self, in_channels=2, conv_out=256, capsule_dim=8, num_capsules=32, num_classes=8, img_size=(224, 224)):
        super(CustomCapsuleNet, self).__init__()
        self.num_classes = num_classes

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels, conv_out, kernel_size=9, stride=1, padding=0)

        # Primary Capsules
        self.primary_caps = nn.Conv2d(conv_out, num_capsules * capsule_dim, kernel_size=9, stride=2, padding=0)

        # Compute capsule input size dynamically
        conv1_size = conv_output_size(img_size[0], 9, 1, 0)
        capsule_input_size = conv_output_size(conv1_size, 9, 2, 0)

        # Digit Capsules
        self.digit_caps = nn.Linear(num_capsules * capsule_dim * capsule_input_size * capsule_input_size,
                                    num_classes * 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        x = x.view(x.size(0), -1)
        x = self.digit_caps(x)
        x = x.view(x.size(0), self.num_classes, 16)  # Use self.num_classes from __init__
        return x.norm(dim=-1)  # Get class probabilities


class DeepCapsuleNet(nn.Module):
    def __init__(self, in_channels=2, conv1_out=128, conv2_out=256, capsule_dim=8, num_capsules=32, num_classes=8,
                 img_size=(224, 224)):
        super(DeepCapsuleNet, self).__init__()
        self.num_classes = num_classes

        # Two Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5, stride=1, padding=2)

        # Primary Capsules
        self.primary_caps = nn.Conv2d(conv2_out, num_capsules * capsule_dim, kernel_size=9, stride=2, padding=0)

        # Compute capsule input size dynamically
        conv1_size = conv_output_size(img_size[0], 5, 1, 2)
        conv2_size = conv_output_size(conv1_size, 5, 1, 2)
        capsule_input_size = conv_output_size(conv2_size, 9, 2, 0)

        # Digit Capsules
        self.digit_caps = nn.Linear(num_capsules * capsule_dim * capsule_input_size * capsule_input_size,
                                    num_classes * 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.primary_caps(x)
        x = x.view(x.size(0), -1)
        x = self.digit_caps(x)
        x = x.view(x.size(0), self.num_classes, 16)  # Use self.num_classes from __init__
        return x.norm(dim=-1)  # Get class probabilities
