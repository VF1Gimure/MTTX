import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)  # DenseNet-style concatenation

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return F.avg_pool2d(x, 2)  # Downsampling

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        layers = [BasicBlock(in_channels + i * growth_rate, growth_rate, drop_rate) for i in range(num_layers)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DenseNet(nn.Module):
    def __init__(self, in_channels=2, out_features=8, growth_rate=32, num_layers=3, reduction=0.5, drop_rate=0.0):
        super(DenseNet, self).__init__()
        self.init_channels = 2 * growth_rate
        block_layers = num_layers // 3  # Split layers across 3 Dense Blocks

        # Initial Convolution
        self.conv1 = nn.Conv2d(in_channels, self.init_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # First Dense Block + Transition
        self.block1 = DenseBlock(block_layers, self.init_channels, growth_rate, drop_rate)
        in_channels = self.init_channels + block_layers * growth_rate
        self.trans1 = TransitionBlock(in_channels, int(in_channels * reduction), drop_rate)
        in_channels = int(in_channels * reduction)

        # Second Dense Block + Transition
        self.block2 = DenseBlock(block_layers, in_channels, growth_rate, drop_rate)
        in_channels += block_layers * growth_rate
        self.trans2 = TransitionBlock(in_channels, int(in_channels * reduction), drop_rate)
        in_channels = int(in_channels * reduction)

        # Third Dense Block
        self.block3 = DenseBlock(block_layers, in_channels, growth_rate, drop_rate)
        in_channels += block_layers * growth_rate

        # Final BatchNorm + Global Pooling + FC Layer
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.block3(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)  # Global pooling
        x = torch.flatten(x, 1)
        return self.fc(x)



class CustomDenseNet(nn.Module):
    def __init__(self, in_channels=2, out_features=8, growth_rate=32, num_layers=3, img_size=(224,224)):
        super(CustomDenseNet, self).__init__()
        self.init_channels = 2 * growth_rate
        drop_rate = 0.0
        # Initial Convolution
        self.conv1 = nn.Conv2d(in_channels, self.init_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Single Dense Block
        self.block = DenseBlock(num_layers, self.init_channels, growth_rate, drop_rate)

        # Final BatchNorm + Global Pooling + FC Layer
        in_channels = self.init_channels + num_layers * growth_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1)  # Global pooling to reduce feature map size
        x = torch.flatten(x, 1)
        return self.fc(x)