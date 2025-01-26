import torch.nn as nn
import torch.nn.functional as F
conv_k3 = lambda in_channels, out_channels: nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv = conv_k3(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.bn(self.conv(x))))  # Conv1 + BatchNorm + Pool
        return self.dropout(x)

class CNN3(nn.Module):
    def __init__(self, in_channels, channels, dropout_rates, out_features, multiplier):
        super().__init__()
        # assert len(channels) == len(dropout_rates) == 5, "5 canales y dropout rates."

        # Feature extraction layers
        self.features = nn.Sequential(
            ConvBlock(in_channels, channels[0], dropout_rates[0]),
            ConvBlock(channels[0], channels[1], dropout_rates[1]),
            ConvBlock(channels[1], channels[2], dropout_rates[2]),
            ConvBlock(channels[2], channels[3], dropout_rates[3]),
            # ConvBlock(channels[3], channels[4], dropout_rates[4])
        )

        # Classifier layers
        self.classifier = nn.Sequential(  # 4 si no se borra el ultimo
            FullyConnectedBlock(channels[3] * multiplier, channels[2], dropout_rates[2]),
            FullyConnectedBlock(channels[2], channels[1], dropout_rates[1]),
            nn.Linear(channels[1], out_features),
            nn.Softmax(dim=1)  # Final classification layer
        )

    def forward(self, x):
        x = self.features(x)
        print(f"Output shape after features: {x.shape}")  # Debug the shape
        x = self.classifier(x)
        return x