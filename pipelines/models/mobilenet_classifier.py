import torchvision.models as models
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(MobileNetClassifier, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)
