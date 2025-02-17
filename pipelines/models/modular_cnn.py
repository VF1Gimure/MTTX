from torch import nn

from pipelines.models.cnn_custom import CNN_custom


class CustomCNN(nn.Module):
    def __init__(self, in_channels=2, channel1=32, channel2=64, out_features=8, img_size=(512, 512)):
        super().__init__()
        self.conv1 = CNN_custom(in_channels, channel1, channel1)
        self.conv2 = CNN_custom(channel1, channel2, channel2)
        self.flatten = nn.Flatten()

        # Compute final flattened size
        flattened_size = channel2 * (img_size[0] // 4) * (img_size[1] // 4)  # Since each CNN_custom does 2x2 MaxPool

        self.fc = nn.Linear(flattened_size, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.fc(x)



class CustomCNN_3(nn.Module):
    def __init__(self, in_channels=2, channel1=32, channel2=64, channel3=128, out_features=8, img_size=(512, 512)):
        super().__init__()
        self.conv1 = CNN_custom(in_channels, channel1, channel1)
        self.conv2 = CNN_custom(channel1, channel2, channel2)
        self.conv3 = CNN_custom(channel2, channel2, channel3)
        self.flatten = nn.Flatten()

        # Compute final flattened size
        flattened_size = channel3 * (img_size[0] // 8) * (img_size[1] // 8)  # Since each CNN_custom does 2x2 MaxPool

        self.fc = nn.Linear(flattened_size, out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.fc(x)